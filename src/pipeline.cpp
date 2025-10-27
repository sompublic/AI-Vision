/**
 * @file pipeline.cpp
 * @brief Async multi-threaded pipeline for PlowPilot AI-Vision
 * @author PlowPilot Team
 * @date 2025-01-25
 * 
 * P2: Async producer/consumer pipeline with bounded ring buffers
 * Target: 720p @ 30 FPS end-to-end with p95 latency ≤80 ms
 * Stable GPU util 60–90%, no queue growth (bounded latency)
 */

#define INCLUDED_IN_PIPELINE
#include "capture_gst.cpp"
#include "infer_trt.cpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#ifdef USE_ALTERNATIVE_YAML
#include "stub_yaml.h"
#else
#include <yaml-cpp/yaml.h>
#endif

struct FrameData {
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp;
    uint64_t frame_id;
    
    FrameData() : frame_id(0) {
        timestamp = std::chrono::steady_clock::now();
    }
    
    FrameData(const cv::Mat& f, uint64_t id) : frame(f.clone()), frame_id(id) {
        timestamp = std::chrono::steady_clock::now();
    }
};

struct DetectionData {
    std::vector<Detection> detections;
    std::chrono::steady_clock::time_point timestamp;
    uint64_t frame_id;
    
    DetectionData() : frame_id(0) {
        timestamp = std::chrono::steady_clock::now();
    }
    
    DetectionData(const std::vector<Detection>& dets, uint64_t id) : detections(dets), frame_id(id) {
        timestamp = std::chrono::steady_clock::now();
    }
};

class AsyncPipeline {
private:
    // Components
    std::unique_ptr<GStreamerCapture> capture_;
    std::unique_ptr<TensorRTInference> inference_;
    
    // Threads
    std::thread capture_thread_;
    std::thread inference_thread_;
    std::thread display_thread_;
    std::thread recording_thread_;
    
    // Queues with bounded buffers
    std::queue<FrameData> capture_queue_;
    std::queue<FrameData> inference_queue_;
    std::queue<DetectionData> display_queue_;
    std::queue<FrameData> recording_queue_;
    
    // Queue mutexes and conditions
    std::mutex capture_mutex_, inference_mutex_, display_mutex_, recording_mutex_;
    std::condition_variable capture_condition_, inference_condition_, 
                           display_condition_, recording_condition_;
    
    // Queue sizes
    size_t capture_queue_size_, inference_queue_size_, 
           display_queue_size_, recording_queue_size_;
    
    // Control
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;
    
    // Performance monitoring
    std::atomic<uint64_t> total_frames_;
    std::atomic<uint64_t> processed_frames_;
    std::atomic<uint64_t> dropped_frames_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Frame ID counter
    std::atomic<uint64_t> frame_counter_;
    
    // Configuration
    std::string drop_policy_;
    int max_queue_wait_;
    bool display_enabled_;
    bool recording_enabled_;
    
public:
    AsyncPipeline() : capture_queue_size_(4), inference_queue_size_(2),
                     display_queue_size_(2), recording_queue_size_(8),
                     running_(false), initialized_(false),
                     total_frames_(0), processed_frames_(0), dropped_frames_(0),
                     frame_counter_(0), drop_policy_("oldest"),
                     max_queue_wait_(100), display_enabled_(true),
                     recording_enabled_(false) {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    ~AsyncPipeline() {
        stop();
    }
    
    bool initialize(const std::string& config_file) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            auto pipeline = config["pipeline"];
            
            // Load queue sizes
            auto queues = pipeline["queues"];
            capture_queue_size_ = queues["capture_queue_size"].as<size_t>(4);
            inference_queue_size_ = queues["inference_queue_size"].as<size_t>(2);
            display_queue_size_ = queues["display_queue_size"].as<size_t>(2);
            recording_queue_size_ = queues["recording_queue_size"].as<size_t>(8);
            
            // Load control settings
            drop_policy_ = queues["drop_policy"].as<std::string>("oldest");
            max_queue_wait_ = queues["max_queue_wait"].as<int>(100);
            
            // Load feature flags
            display_enabled_ = pipeline["display"]["enabled"].as<bool>(true);
            recording_enabled_ = pipeline["recording"]["enabled"].as<bool>(false);
            
            // Initialize components
            capture_ = std::make_unique<GStreamerCapture>();
            inference_ = std::make_unique<TensorRTInference>();
            
            if (!capture_->initialize("configs/camera.yaml")) {
                std::cerr << "Failed to initialize capture" << std::endl;
                return false;
            }
            
            if (!inference_->initialize("configs/model.yaml")) {
                std::cerr << "Failed to initialize inference" << std::endl;
                return false;
            }
            
            initialized_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool start() {
        if (!initialized_) {
            std::cerr << "Pipeline not initialized" << std::endl;
            return false;
        }
        
        // Start components
        if (!capture_->start()) {
            std::cerr << "Failed to start capture" << std::endl;
            return false;
        }
        
        if (!inference_->start()) {
            std::cerr << "Failed to start inference" << std::endl;
            return false;
        }
        
        // Start pipeline threads
        running_ = true;
        capture_thread_ = std::thread(&AsyncPipeline::captureLoop, this);
        inference_thread_ = std::thread(&AsyncPipeline::inferenceLoop, this);
        
        if (display_enabled_) {
            display_thread_ = std::thread(&AsyncPipeline::displayLoop, this);
        }
        
        if (recording_enabled_) {
            recording_thread_ = std::thread(&AsyncPipeline::recordingLoop, this);
        }
        
        std::cout << "Async pipeline started" << std::endl;
        return true;
    }
    
    void stop() {
        running_ = false;
        
        // Notify all threads
        capture_condition_.notify_all();
        inference_condition_.notify_all();
        display_condition_.notify_all();
        recording_condition_.notify_all();
        
        // Stop components
        if (capture_) capture_->stop();
        if (inference_) inference_->stop();
        
        // Wait for threads
        if (capture_thread_.joinable()) capture_thread_.join();
        if (inference_thread_.joinable()) inference_thread_.join();
        if (display_thread_.joinable()) display_thread_.join();
        if (recording_thread_.joinable()) recording_thread_.join();
    }
    
    // Performance metrics
    double getFPS() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        if (duration > 0) {
            return static_cast<double>(processed_frames_.load()) / duration;
        }
        return 0.0;
    }
    
    uint64_t getTotalFrames() const { return total_frames_.load(); }
    uint64_t getProcessedFrames() const { return processed_frames_.load(); }
    uint64_t getDroppedFrames() const { return dropped_frames_.load(); }
    bool isRunning() const { return running_.load(); }
    
private:
    void captureLoop() {
        cv::Mat frame;
        
        while (running_) {
            if (capture_->getFrame(frame, 100)) {
                uint64_t frame_id = frame_counter_++;
                FrameData frame_data(frame, frame_id);
                
                // Add to capture queue
                {
                    std::lock_guard<std::mutex> lock(capture_mutex_);
                    if (capture_queue_.size() >= capture_queue_size_) {
                        if (drop_policy_ == "oldest") {
                            capture_queue_.pop();
                            dropped_frames_++;
                        }
                    }
                    capture_queue_.push(frame_data);
                }
                capture_condition_.notify_one();
                
                // Add to inference queue
                {
                    std::lock_guard<std::mutex> lock(inference_mutex_);
                    if (inference_queue_.size() >= inference_queue_size_) {
                        if (drop_policy_ == "oldest") {
                            inference_queue_.pop();
                        }
                    }
                    inference_queue_.push(frame_data);
                }
                inference_condition_.notify_one();
                
                // Add to recording queue if enabled
                if (recording_enabled_) {
                    std::lock_guard<std::mutex> lock(recording_mutex_);
                    if (recording_queue_.size() >= recording_queue_size_) {
                        if (drop_policy_ == "oldest") {
                            recording_queue_.pop();
                        }
                    }
                    recording_queue_.push(frame_data);
                }
                recording_condition_.notify_one();
                
                total_frames_++;
            }
        }
    }
    
    void inferenceLoop() {
        while (running_) {
            FrameData frame_data;
            
            // Get frame from inference queue
            {
                std::unique_lock<std::mutex> lock(inference_mutex_);
                if (inference_condition_.wait_for(lock, std::chrono::milliseconds(max_queue_wait_),
                                               [this] { return !inference_queue_.empty() || !running_; })) {
                    if (!inference_queue_.empty()) {
                        frame_data = inference_queue_.front();
                        inference_queue_.pop();
                    }
                }
            }
            
            if (frame_data.frame.empty()) {
                continue;
            }
            
            // Add frame to inference
            inference_->addFrame(frame_data.frame);
            
            // Get detections from inference
            std::vector<Detection> detections;
            if (inference_->getDetections(detections, 100)) {
                DetectionData detection_data(detections, frame_data.frame_id);
                
                // Add to display queue
                if (display_enabled_) {
                    std::lock_guard<std::mutex> lock(display_mutex_);
                    if (display_queue_.size() >= display_queue_size_) {
                        if (drop_policy_ == "oldest") {
                            display_queue_.pop();
                        }
                    }
                    display_queue_.push(detection_data);
                }
                display_condition_.notify_one();
                
                processed_frames_++;
            }
        }
    }
    
    void displayLoop() {
        cv::Mat display_frame;
        
        while (running_) {
            DetectionData detection_data;
            
            // Get detections from display queue
            {
                std::unique_lock<std::mutex> lock(display_mutex_);
                if (display_condition_.wait_for(lock, std::chrono::milliseconds(max_queue_wait_),
                                               [this] { return !display_queue_.empty() || !running_; })) {
                    if (!display_queue_.empty()) {
                        detection_data = display_queue_.front();
                        display_queue_.pop();
                    }
                }
            }
            
            if (detection_data.detections.empty()) {
                continue;
            }
            
            // Get corresponding frame from capture queue
            {
                std::lock_guard<std::mutex> lock(capture_mutex_);
                if (!capture_queue_.empty()) {
                    display_frame = capture_queue_.front().frame.clone();
                    capture_queue_.pop();
                }
            }
            
            if (display_frame.empty()) {
                continue;
            }
            
            // Draw detections
            drawDetections(display_frame, detection_data.detections);
            
            // Show frame
            cv::imshow("PlowPilot AI-Vision", display_frame);
            
            if (cv::waitKey(1) == 'q') {
                running_ = false;
                break;
            }
        }
    }
    
    void recordingLoop() {
        // TODO: Implement recording functionality
        // This will be implemented in P3
        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            // Draw bounding box
            cv::Rect rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
            
            // Draw label
            std::string label = det.class_name + " " + std::to_string(det.confidence);
            cv::putText(frame, label, cv::Point(det.x1, det.y1 - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        
        // Draw FPS
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(getFPS()));
        cv::putText(frame, fps_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    }
};

// Example usage and testing
int main(int argc, char* argv[]) {
    std::string config_file = "configs/pipeline.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    AsyncPipeline pipeline;
    
    if (!pipeline.initialize(config_file)) {
        std::cerr << "Failed to initialize pipeline" << std::endl;
        return -1;
    }
    
    if (!pipeline.start()) {
        std::cerr << "Failed to start pipeline" << std::endl;
        return -1;
    }
    
    std::cout << "Pipeline running. Press 'q' to quit." << std::endl;
    
    // Monitor performance
    auto start_time = std::chrono::steady_clock::now();
    auto last_stats_time = start_time;
    
    while (pipeline.isRunning()) {
        auto now = std::chrono::steady_clock::now();
        
        // Print stats every 10 seconds
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count() >= 10) {
            std::cout << "FPS: " << pipeline.getFPS()
                      << ", Total: " << pipeline.getTotalFrames()
                      << ", Processed: " << pipeline.getProcessedFrames()
                      << ", Dropped: " << pipeline.getDroppedFrames() << std::endl;
            last_stats_time = now;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    pipeline.stop();
    cv::destroyAllWindows();
    
    std::cout << "Final stats:" << std::endl;
    std::cout << "Total frames: " << pipeline.getTotalFrames() << std::endl;
    std::cout << "Processed frames: " << pipeline.getProcessedFrames() << std::endl;
    std::cout << "Dropped frames: " << pipeline.getDroppedFrames() << std::endl;
    std::cout << "Average FPS: " << pipeline.getFPS() << std::endl;
    
    return 0;
}
