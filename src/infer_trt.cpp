/**
 * @file infer_trt.cpp
 * @brief TensorRT inference wrapper for PlowPilot AI-Vision
 * @author PlowPilot Team
 * @date 2025-01-25
 * 
 * P2: TensorRT inference with async pipeline
 * Target: 720p @ 30 FPS end-to-end with p95 latency ≤80 ms
 * Model: YOLOv8n FP16 (target infer time ≤20–25 ms)
 * GPU util: 60–90%, no queue growth (bounded latency)
 */

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <fstream>
#ifdef USE_ALTERNATIVE_YAML
#include "stub_yaml.h"
#else
#include <yaml-cpp/yaml.h>
#endif

struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float confidence;      // Detection confidence
    int class_id;          // Class ID
    std::string class_name; // Class name
};

// Simple logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TensorRTInference {
private:
    Logger logger_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    // Input/output configuration
    int input_width_, input_height_, input_channels_;
    int output_size_;
    void* d_input_;
    void* d_output_;
    
    // Performance monitoring
    std::atomic<uint64_t> inference_count_;
    std::atomic<double> total_inference_time_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Async processing
    std::queue<cv::Mat> input_queue_;
    std::queue<std::vector<Detection>> output_queue_;
    std::mutex input_mutex_, output_mutex_;
    std::condition_variable input_condition_, output_condition_;
    std::atomic<bool> running_;
    std::thread inference_thread_;
    
    // Configuration
    float confidence_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;
    
public:
    TensorRTInference() : runtime_(nullptr), engine_(nullptr), context_(nullptr),
                         input_width_(640), input_height_(640), input_channels_(3),
                         output_size_(25200 * 85), d_input_(nullptr), d_output_(nullptr),
                         inference_count_(0), total_inference_time_(0.0),
                         confidence_threshold_(0.5), nms_threshold_(0.45),
                         running_(false) {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    ~TensorRTInference() {
        stop();
        cleanup();
    }
    
    bool initialize(const std::string& config_file) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            auto model = config["model"];
            
            // Load model configuration
            auto input = model["input"];
            input_width_ = input["width"].as<int>(640);
            input_height_ = input["height"].as<int>(640);
            input_channels_ = input["channels"].as<int>(3);
            
            auto output = model["output"];
            confidence_threshold_ = output["confidence_threshold"].as<float>(0.5);
            nms_threshold_ = output["nms_threshold"].as<float>(0.45);
            
            // Load class names
            if (config["classes"]) {
                for (const auto& class_name : config["classes"]) {
                    class_names_.push_back(class_name.as<std::string>());
                }
            }
            
            // Load TensorRT engine
            std::string engine_path = model["trt_engine_path"].as<std::string>("models/yolov8n_fp16.trt");
            if (!loadEngine(engine_path)) {
                std::cerr << "Failed to load TensorRT engine: " << engine_path << std::endl;
                return false;
            }
            
            // Allocate GPU memory
            if (!allocateMemory()) {
                std::cerr << "Failed to allocate GPU memory" << std::endl;
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool loadEngine(const std::string& engine_path) {
        // Create runtime
        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        // Load engine from file
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Failed to open engine file: " << engine_path << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
        
        // Deserialize engine
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
        if (!engine_) {
            std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
            return false;
        }
        
        // Create execution context
        context_ = engine_->createExecutionContext();
        if (!context_) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        std::cout << "TensorRT engine loaded successfully" << std::endl;
        return true;
    }
    
    bool allocateMemory() {
        // Calculate input/output sizes
        size_t input_size = input_width_ * input_height_ * input_channels_ * sizeof(float);
        size_t output_size = output_size_ * sizeof(float);
        
        // Allocate GPU memory (simplified - in real implementation, use CUDA)
        // For now, we'll use CPU memory as a placeholder
        d_input_ = malloc(input_size);
        d_output_ = malloc(output_size);
        
        if (!d_input_ || !d_output_) {
            std::cerr << "Failed to allocate memory" << std::endl;
            return false;
        }
        
        std::cout << "Memory allocated: input=" << input_size << " bytes, output=" << output_size << " bytes" << std::endl;
        return true;
    }
    
    bool start() {
        running_ = true;
        inference_thread_ = std::thread(&TensorRTInference::inferenceLoop, this);
        std::cout << "TensorRT inference started" << std::endl;
        return true;
    }
    
    void stop() {
        running_ = false;
        input_condition_.notify_all();
        output_condition_.notify_all();
        
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
        
        std::cout << "TensorRT inference stopped" << std::endl;
    }
    
    void addFrame(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(input_mutex_);
        if (input_queue_.size() < 4) { // Limit queue size
            input_queue_.push(frame.clone());
        }
        input_condition_.notify_one();
    }
    
    bool getDetections(std::vector<Detection>& detections, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(output_mutex_);
        if (output_condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                     [this] { return !output_queue_.empty(); })) {
            if (!output_queue_.empty()) {
                detections = output_queue_.front();
                output_queue_.pop();
                return true;
            }
        }
        return false;
    }
    
    // Performance metrics
    double getInferenceTime() const {
        uint64_t count = inference_count_.load();
        if (count > 0) {
            return total_inference_time_.load() / count;
        }
        return 0.0;
    }
    
    uint64_t getInferenceCount() const { return inference_count_.load(); }
    double getFPS() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        if (duration > 0) {
            return static_cast<double>(inference_count_.load()) / duration;
        }
        return 0.0;
    }
    
private:
    void inferenceLoop() {
        while (running_) {
            cv::Mat frame;
            
            // Get frame from input queue
            {
                std::unique_lock<std::mutex> lock(input_mutex_);
                if (input_condition_.wait_for(lock, std::chrono::milliseconds(100),
                                             [this] { return !input_queue_.empty() || !running_; })) {
                    if (!input_queue_.empty()) {
                        frame = input_queue_.front();
                        input_queue_.pop();
                    }
                }
            }
            
            if (frame.empty()) {
                continue;
            }
            
            // Perform inference
            auto start_time = std::chrono::steady_clock::now();
            std::vector<Detection> detections = performInference(frame);
            auto end_time = std::chrono::steady_clock::now();
            
            auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
            
            // Update statistics
            inference_count_++;
            total_inference_time_ = total_inference_time_.load() + inference_time;
            
            // Add detections to output queue
            {
                std::lock_guard<std::mutex> lock(output_mutex_);
                if (output_queue_.size() < 4) { // Limit queue size
                    output_queue_.push(detections);
                }
            }
            output_condition_.notify_one();
        }
    }
    
    std::vector<Detection> performInference(const cv::Mat& frame) {
        // Preprocess frame
        cv::Mat resized, normalized;
        cv::resize(frame, resized, cv::Size(input_width_, input_height_));
        resized.convertTo(normalized, CV_32F, 1.0/255.0);
        
        // Copy to input buffer (simplified - in real implementation, use CUDA)
        memcpy(d_input_, normalized.data, input_width_ * input_height_ * input_channels_ * sizeof(float));
        
        // Perform inference (simplified - in real implementation, use TensorRT)
        // For now, we'll simulate inference with dummy detections
        std::vector<Detection> detections;
        
        // Simulate some detections for testing
        if (inference_count_.load() % 30 == 0) { // Every 30 frames
            Detection det;
            det.x1 = 100; det.y1 = 100; det.x2 = 200; det.y2 = 200;
            det.confidence = 0.8f;
            det.class_id = 0;
            det.class_name = "person";
            detections.push_back(det);
        }
        
        return detections;
    }
    
    void cleanup() {
        if (d_input_) {
            free(d_input_);
            d_input_ = nullptr;
        }
        if (d_output_) {
            free(d_output_);
            d_output_ = nullptr;
        }
        if (context_) {
            delete context_;
            context_ = nullptr;
        }
        if (engine_) {
            delete engine_;
            engine_ = nullptr;
        }
        if (runtime_) {
            delete runtime_;
            runtime_ = nullptr;
        }
    }
};

// Test function for standalone testing
int test_inference_main(int argc, char* argv[]) {
    std::string config_file = "configs/model.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    TensorRTInference inference;
    
    if (!inference.initialize(config_file)) {
        std::cerr << "Failed to initialize TensorRT inference" << std::endl;
        return -1;
    }
    
    if (!inference.start()) {
        std::cerr << "Failed to start TensorRT inference" << std::endl;
        return -1;
    }
    
    // Test inference for 30 seconds
    std::cout << "Testing TensorRT inference for 30 seconds..." << std::endl;
    
    auto start_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    
    while (std::chrono::duration_cast<std::chrono::seconds>(
           std::chrono::steady_clock::now() - start_time).count() < 30) {
        
        // Create test frame
        cv::Mat test_frame = cv::Mat::zeros(640, 640, CV_8UC3);
        cv::putText(test_frame, "Test Frame " + std::to_string(frame_count++),
                   cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 2.0,
                   cv::Scalar(255, 255, 255), 2);
        
        // Add frame to inference
        inference.addFrame(test_frame);
        
        // Check for detections
        std::vector<Detection> detections;
        if (inference.getDetections(detections, 100)) {
            std::cout << "Detections: " << detections.size() << std::endl;
            for (const auto& det : detections) {
                std::cout << "  " << det.class_name << " (confidence: " << det.confidence << ")" << std::endl;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }
    
    inference.stop();
    
    std::cout << "Inference test completed:" << std::endl;
    std::cout << "Total inferences: " << inference.getInferenceCount() << std::endl;
    std::cout << "Average inference time: " << inference.getInferenceTime() << " ms" << std::endl;
    std::cout << "Average FPS: " << inference.getFPS() << std::endl;
    
    return 0;
}

// Main function for test_inference executable
#ifndef INCLUDED_IN_PIPELINE
int main(int argc, char* argv[]) {
    return test_inference_main(argc, argv);
}
#endif