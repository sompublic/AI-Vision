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
#include <yaml-cpp/yaml.h>

struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float confidence;      // Detection confidence
    int class_id;          // Class ID
    std::string class_name; // Class name
};

class TensorRTInference {
private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    // Model configuration
    int input_width_, input_height_, input_channels_;
    int batch_size_;
    std::string precision_;
    
    // GPU memory
    void* d_input_;
    void* d_output_;
    size_t input_size_, output_size_;
    
    // Performance monitoring
    std::atomic<uint64_t> inference_count_;
    std::atomic<double> total_inference_time_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Detection configuration
    float confidence_threshold_;
    float nms_threshold_;
    int max_detections_;
    std::vector<std::string> class_names_;
    
    // Async processing
    std::queue<cv::Mat> input_queue_;
    std::queue<std::vector<Detection>> output_queue_;
    std::mutex input_mutex_, output_mutex_;
    std::condition_variable input_condition_, output_condition_;
    std::atomic<bool> running_;
    std::thread inference_thread_;
    
public:
    TensorRTInference() : runtime_(nullptr), engine_(nullptr), context_(nullptr),
                         d_input_(nullptr), d_output_(nullptr),
                         input_width_(640), input_height_(640), input_channels_(3),
                         batch_size_(1), precision_("fp16"),
                         inference_count_(0), total_inference_time_(0.0),
                         confidence_threshold_(0.5), nms_threshold_(0.45),
                         max_detections_(100), running_(false) {
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
            std::string engine_path = model["trt_engine_path"].as<std::string>("models/yolov8n_fp16.trt");
            input_width_ = model["input"]["width"].as<int>(640);
            input_height_ = model["input"]["height"].as<int>(640);
            input_channels_ = model["input"]["channels"].as<int>(3);
            batch_size_ = model["input"]["batch_size"].as<int>(1);
            precision_ = model["input"]["data_type"].as<std::string>("fp16");
            
            // Load detection configuration
            confidence_threshold_ = model["output"]["confidence_threshold"].as<float>(0.5);
            nms_threshold_ = model["output"]["nms_threshold"].as<float>(0.45);
            max_detections_ = model["output"]["max_detections"].as<int>(100);
            
            // Load class names
            if (model["classes"]) {
                for (const auto& class_name : model["classes"]) {
                    class_names_.push_back(class_name.as<std::string>());
                }
            }
            
            return loadEngine(engine_path);
        } catch (const std::exception& e) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool loadEngine(const std::string& engine_path) {
        // Initialize TensorRT
        runtime_ = nvinfer1::createInferRuntime(gLogger);
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
            std::cerr << "Failed to deserialize engine" << std::endl;
            return false;
        }
        
        // Create execution context
        context_ = engine_->createExecutionContext();
        if (!context_) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // Allocate GPU memory
        if (!allocateMemory()) {
            return false;
        }
        
        std::cout << "TensorRT engine loaded successfully" << std::endl;
        return true;
    }
    
    bool allocateMemory() {
        // Get input/output dimensions
        auto input_dims = engine_->getBindingDimensions(0);
        auto output_dims = engine_->getBindingDimensions(1);
        
        input_size_ = batch_size_ * input_channels_ * input_height_ * input_width_ * sizeof(float);
        output_size_ = batch_size_ * output_dims.d[1] * output_dims.d[2] * sizeof(float);
        
        // Allocate GPU memory
        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_output_, output_size_);
        
        if (!d_input_ || !d_output_) {
            std::cerr << "Failed to allocate GPU memory" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool start() {
        if (!engine_ || !context_) {
            std::cerr << "Inference not initialized" << std::endl;
            return false;
        }
        
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
    }
    
    bool infer(const cv::Mat& frame, std::vector<Detection>& detections) {
        if (!running_) {
            return false;
        }
        
        // Add frame to input queue
        {
            std::lock_guard<std::mutex> lock(input_mutex_);
            input_queue_.push(frame.clone());
        }
        input_condition_.notify_one();
        
        // Wait for results
        std::unique_lock<std::mutex> lock(output_mutex_);
        if (output_condition_.wait_for(lock, std::chrono::milliseconds(100),
                                      [this] { return !output_queue_.empty() || !running_; })) {
            if (!output_queue_.empty()) {
                detections = output_queue_.front();
                output_queue_.pop();
                return true;
            }
        }
        return false;
    }
    
    // Performance metrics
    double getAverageInferenceTime() const {
        if (inference_count_.load() > 0) {
            return total_inference_time_.load() / inference_count_.load();
        }
        return 0.0;
    }
    
    uint64_t getInferenceCount() const { return inference_count_.load(); }
    bool isRunning() const { return running_.load(); }
    
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
            
            // Preprocess frame
            cv::Mat preprocessed = preprocessFrame(frame);
            
            // Run inference
            auto start_time = std::chrono::high_resolution_clock::now();
            std::vector<Detection> detections = runInference(preprocessed);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            // Update performance metrics
            auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
            total_inference_time_ += inference_time;
            inference_count_++;
            
            // Add results to output queue
            {
                std::lock_guard<std::mutex> lock(output_mutex_);
                output_queue_.push(detections);
            }
            output_condition_.notify_one();
        }
    }
    
    cv::Mat preprocessFrame(const cv::Mat& frame) {
        cv::Mat resized, normalized;
        
        // Resize to model input size
        cv::resize(frame, resized, cv::Size(input_width_, input_height_));
        
        // Convert BGR to RGB and normalize
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(normalized, CV_32F, 1.0/255.0);
        
        return normalized;
    }
    
    std::vector<Detection> runInference(const cv::Mat& preprocessed) {
        // Copy input to GPU
        cudaMemcpy(d_input_, preprocessed.data, input_size_, cudaMemcpyHostToDevice);
        
        // Run inference
        void* bindings[] = {d_input_, d_output_};
        bool success = context_->executeV2(bindings);
        
        if (!success) {
            std::cerr << "TensorRT inference failed" << std::endl;
            return {};
        }
        
        // Copy output from GPU
        std::vector<float> output(output_size_ / sizeof(float));
        cudaMemcpy(output.data(), d_output_, output_size_, cudaMemcpyDeviceToHost);
        
        // Post-process detections
        return postprocessDetections(output);
    }
    
    std::vector<Detection> postprocessDetections(const std::vector<float>& output) {
        std::vector<Detection> detections;
        
        // YOLOv8 output format: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
        int num_detections = output.size() / (4 + 80); // 8400 for YOLOv8n
        
        for (int i = 0; i < num_detections; i++) {
            // Get bounding box coordinates
            float x_center = output[i * 84 + 0];
            float y_center = output[i * 84 + 1];
            float width = output[i * 84 + 2];
            float height = output[i * 84 + 3];
            
            // Find class with highest confidence
            float max_confidence = 0.0;
            int best_class = -1;
            for (int j = 0; j < 80; j++) {
                float confidence = output[i * 84 + 4 + j];
                if (confidence > max_confidence) {
                    max_confidence = confidence;
                    best_class = j;
                }
            }
            
            // Apply confidence threshold
            if (max_confidence > confidence_threshold_) {
                Detection det;
                det.x1 = x_center - width / 2;
                det.y1 = y_center - height / 2;
                det.x2 = x_center + width / 2;
                det.y2 = y_center + height / 2;
                det.confidence = max_confidence;
                det.class_id = best_class;
                det.class_name = (best_class < class_names_.size()) ? class_names_[best_class] : "unknown";
                
                detections.push_back(det);
            }
        }
        
        // Apply NMS
        return applyNMS(detections);
    }
    
    std::vector<Detection> applyNMS(const std::vector<Detection>& detections) {
        std::vector<Detection> filtered;
        std::vector<bool> suppressed(detections.size(), false);
        
        // Sort by confidence
        std::vector<std::pair<float, int>> sorted_detections;
        for (int i = 0; i < detections.size(); i++) {
            sorted_detections.push_back({detections[i].confidence, i});
        }
        std::sort(sorted_detections.begin(), sorted_detections.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        for (const auto& pair : sorted_detections) {
            int idx = pair.second;
            if (suppressed[idx]) continue;
            
            filtered.push_back(detections[idx]);
            
            // Suppress overlapping detections
            for (int j = idx + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;
                
                float iou = calculateIoU(detections[idx], detections[j]);
                if (iou > nms_threshold_) {
                    suppressed[j] = true;
                }
            }
        }
        
        return filtered;
    }
    
    float calculateIoU(const Detection& det1, const Detection& det2) {
        float x1 = std::max(det1.x1, det2.x1);
        float y1 = std::max(det1.y1, det2.y1);
        float x2 = std::min(det1.x2, det2.x2);
        float y2 = std::min(det1.y2, det2.y2);
        
        if (x2 <= x1 || y2 <= y1) return 0.0;
        
        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1);
        float area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1);
        float union_area = area1 + area2 - intersection;
        
        return intersection / union_area;
    }
    
    void cleanup() {
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        if (runtime_) runtime_->destroy();
    }
};

// Global logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// Example usage and testing
int main(int argc, char* argv[]) {
    std::string config_file = "configs/model.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    TensorRTInference inference;
    
    if (!inference.initialize(config_file)) {
        std::cerr << "Failed to initialize inference" << std::endl;
        return -1;
    }
    
    if (!inference.start()) {
        std::cerr << "Failed to start inference" << std::endl;
        return -1;
    }
    
    // Test with a sample image
    cv::Mat test_image = cv::Mat::zeros(720, 1280, CV_8UC3);
    std::vector<Detection> detections;
    
    std::cout << "Testing inference..." << std::endl;
    
    for (int i = 0; i < 100; i++) {
        if (inference.infer(test_image, detections)) {
            std::cout << "Inference " << i << ": " << detections.size() << " detections" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }
    
    inference.stop();
    
    std::cout << "Inference stats:" << std::endl;
    std::cout << "Total inferences: " << inference.getInferenceCount() << std::endl;
    std::cout << "Average inference time: " << inference.getAverageInferenceTime() << " ms" << std::endl;
    
    return 0;
}
