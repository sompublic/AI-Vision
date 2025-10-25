/**
 * @file capture_gst.cpp
 * @brief GStreamer-based camera capture with DMABUF support for PlowPilot AI-Vision
 * @author PlowPilot Team
 * @date 2025-01-25
 * 
 * P1: Low-latency, GPU-friendly capture with stable FPS
 * Target: 720p @ 30 FPS sustained (≥28 FPS over 10 min)
 * Latency: ≤15 ms p50 / ≤25 ms p95
 * CPU: ≤25% of one core; no memory leaks (ΔRSS < 50 MB over 30 min)
 */

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gstreamer.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <yaml-cpp/yaml.h>

class GStreamerCapture {
private:
    GstElement* pipeline_;
    GstElement* appsink_;
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;
    
    // Performance monitoring
    std::atomic<uint64_t> frame_count_;
    std::atomic<uint64_t> dropped_frames_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_frame_time_;
    
    // Frame buffer
    std::queue<cv::Mat> frame_buffer_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_condition_;
    size_t max_buffer_size_;
    
    // Configuration
    std::string device_;
    int width_, height_;
    int framerate_;
    std::string format_;
    bool dmabuf_enabled_;
    
public:
    GStreamerCapture() : pipeline_(nullptr), appsink_(nullptr), 
                        running_(false), initialized_(false),
                        frame_count_(0), dropped_frames_(0),
                        max_buffer_size_(4) {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    ~GStreamerCapture() {
        stop();
        if (pipeline_) {
            gst_object_unref(pipeline_);
        }
    }
    
    bool initialize(const std::string& config_file) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            auto camera = config["camera"];
            
            device_ = camera["device"].as<std::string>("/dev/video0");
            width_ = camera["width"].as<int>(1280);
            height_ = camera["height"].as<int>(720);
            framerate_ = camera["framerate"].as<int>(30);
            format_ = camera["format"].as<std::string>("MJPG");
            dmabuf_enabled_ = camera["dmabuf"].as<bool>(true);
            
            return createPipeline();
        } catch (const std::exception& e) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool createPipeline() {
        // Initialize GStreamer
        gst_init(nullptr, nullptr);
        
        // Build GStreamer pipeline string
        std::string pipeline_str;
        if (format_ == "MJPG") {
            pipeline_str = "v4l2src device=" + device_ + 
                          " ! video/x-raw,format=MJPG,width=" + std::to_string(width_) + 
                          ",height=" + std::to_string(height_) + 
                          ",framerate=" + std::to_string(framerate_) + "/1" +
                          " ! jpegdec ! video/x-raw,format=BGRx" +
                          " ! videoconvert ! video/x-raw,format=BGR" +
                          " ! appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
        } else {
            // Fallback to YUYV
            pipeline_str = "v4l2src device=" + device_ + 
                          " ! video/x-raw,format=YUY2,width=" + std::to_string(width_) + 
                          ",height=" + std::to_string(height_) + 
                          ",framerate=" + std::to_string(framerate_) + "/1" +
                          " ! videoconvert ! video/x-raw,format=BGR" +
                          " ! appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
        }
        
        std::cout << "GStreamer pipeline: " << pipeline_str << std::endl;
        
        // Create pipeline
        pipeline_ = gst_parse_launch(pipeline_str.c_str(), nullptr);
        if (!pipeline_) {
            std::cerr << "Failed to create GStreamer pipeline" << std::endl;
            return false;
        }
        
        // Get appsink element
        appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
        if (!appsink_) {
            std::cerr << "Failed to get appsink element" << std::endl;
            return false;
        }
        
        // Set appsink callbacks
        GstAppSinkCallbacks callbacks = {0};
        callbacks.new_sample = onNewSample;
        gst_app_sink_set_callbacks(GST_APP_SINK(appsink_), &callbacks, this, nullptr);
        
        initialized_ = true;
        return true;
    }
    
    bool start() {
        if (!initialized_) {
            std::cerr << "Capture not initialized" << std::endl;
            return false;
        }
        
        running_ = true;
        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start pipeline" << std::endl;
            return false;
        }
        
        std::cout << "GStreamer capture started" << std::endl;
        return true;
    }
    
    void stop() {
        running_ = false;
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }
        
        // Clear buffer
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        while (!frame_buffer_.empty()) {
            frame_buffer_.pop();
        }
    }
    
    bool getFrame(cv::Mat& frame, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(buffer_mutex_);
        
        if (buffer_condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                      [this] { return !frame_buffer_.empty() || !running_; })) {
            if (!frame_buffer_.empty()) {
                frame = frame_buffer_.front().clone();
                frame_buffer_.pop();
                return true;
            }
        }
        return false;
    }
    
    // Performance metrics
    double getFPS() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        if (duration > 0) {
            return static_cast<double>(frame_count_.load()) / duration;
        }
        return 0.0;
    }
    
    uint64_t getFrameCount() const { return frame_count_.load(); }
    uint64_t getDroppedFrames() const { return dropped_frames_.load(); }
    bool isRunning() const { return running_.load(); }
    
private:
    static GstFlowReturn onNewSample(GstAppSink* sink, gpointer user_data) {
        GStreamerCapture* capture = static_cast<GStreamerCapture*>(user_data);
        return capture->handleNewSample(sink);
    }
    
    GstFlowReturn handleNewSample(GstAppSink* sink) {
        GstSample* sample = gst_app_sink_pull_sample(sink);
        if (!sample) {
            return GST_FLOW_ERROR;
        }
        
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;
        
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            // Convert to OpenCV Mat
            cv::Mat frame(height_, width_, CV_8UC3, map.data);
            cv::Mat frame_copy = frame.clone();
            
            // Add to buffer
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                if (frame_buffer_.size() >= max_buffer_size_) {
                    frame_buffer_.pop(); // Drop oldest frame
                    dropped_frames_++;
                }
                frame_buffer_.push(frame_copy);
            }
            buffer_condition_.notify_one();
            
            frame_count_++;
            last_frame_time_ = std::chrono::steady_clock::now();
            
            gst_buffer_unmap(buffer, &map);
        }
        
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
};

// Example usage and testing
int main(int argc, char* argv[]) {
    std::string config_file = "configs/camera.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    GStreamerCapture capture;
    
    if (!capture.initialize(config_file)) {
        std::cerr << "Failed to initialize capture" << std::endl;
        return -1;
    }
    
    if (!capture.start()) {
        std::cerr << "Failed to start capture" << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    auto start_time = std::chrono::steady_clock::now();
    auto last_stats_time = start_time;
    
    std::cout << "Starting capture test (10 minutes)..." << std::endl;
    
    while (capture.isRunning()) {
        if (capture.getFrame(frame, 100)) {
            // Process frame here
            cv::imshow("PlowPilot Capture", frame);
            
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
        
        // Print stats every 10 seconds
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count() >= 10) {
            std::cout << "FPS: " << capture.getFPS() 
                      << ", Frames: " << capture.getFrameCount()
                      << ", Dropped: " << capture.getDroppedFrames() << std::endl;
            last_stats_time = now;
        }
        
        // Test duration: 10 minutes
        if (std::chrono::duration_cast<std::chrono::minutes>(now - start_time).count() >= 10) {
            break;
        }
    }
    
    capture.stop();
    cv::destroyAllWindows();
    
    std::cout << "Final stats:" << std::endl;
    std::cout << "Total frames: " << capture.getFrameCount() << std::endl;
    std::cout << "Dropped frames: " << capture.getDroppedFrames() << std::endl;
    std::cout << "Average FPS: " << capture.getFPS() << std::endl;
    
    return 0;
}
