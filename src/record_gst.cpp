/**
 * @file record_gst.cpp
 * @brief GStreamer-based recording for PlowPilot AI-Vision
 * @author PlowPilot Team
 * @date 2025-01-25
 * 
 * P3: Annotated recording with software x264 encoding
 * Target: 720p @ 20–30 FPS without starving inference
 * Alert latency ≤500 ms from event occurrence
 */

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <filesystem>
#include <fstream>
#ifdef USE_ALTERNATIVE_YAML
#include "stub_yaml.h"
#else
#include <yaml-cpp/yaml.h>
#endif

struct RecordingConfig {
    bool enabled;
    bool continuous;
    bool event_based;
    std::string output_dir;
    std::string filename_template;
    int max_file_size;      // MB
    int max_duration;       // seconds
    std::string codec;
    std::string preset;
    int crf;
    int width, height;
    int framerate;
    std::string bitrate;
    
    // Audio settings
    bool audio_enabled;
    std::string audio_device;
    int audio_sample_rate;
    int audio_channels;
    std::string audio_codec;
    int audio_bitrate;
};

class GStreamerRecorder {
private:
    GstElement* pipeline_;
    GstElement* appsrc_;
    GstElement* filesink_;
    
    std::atomic<bool> recording_;
    std::atomic<bool> initialized_;
    std::atomic<bool> event_detected_;
    
    // Recording configuration
    RecordingConfig config_;
    
    // File management
    std::string current_filename_;
    std::chrono::steady_clock::time_point recording_start_time_;
    std::chrono::steady_clock::time_point last_event_time_;
    
    // Performance monitoring
    std::atomic<uint64_t> frames_recorded_;
    std::atomic<uint64_t> files_created_;
    std::chrono::steady_clock::time_point start_time_;
    GstClockTime pipeline_start_time_;
    
    // Event detection
    std::queue<cv::Mat> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::thread recording_thread_;
    
public:
    GStreamerRecorder() : pipeline_(nullptr), appsrc_(nullptr), filesink_(nullptr),
                         recording_(false), initialized_(false), event_detected_(false),
                         frames_recorded_(0), files_created_(0), pipeline_start_time_(0) {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    ~GStreamerRecorder() {
        stop();
        if (pipeline_) {
            gst_object_unref(pipeline_);
        }
    }
    
    bool initialize(const std::string& config_file) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            auto recording = config["recording"];
            
            // Load configuration
            config_.enabled = recording["enabled"].as<bool>(true);
            config_.continuous = recording["continuous"].as<bool>(false);
            config_.event_based = recording["event_based"].as<bool>(true);
            config_.output_dir = recording["output_dir"].as<std::string>("recordings/");
            config_.filename_template = recording["filename_template"].as<std::string>("plowpilot_%Y%m%d_%H%M%S.avi");
            config_.max_file_size = recording["max_file_size"].as<int>(100);
            config_.max_duration = recording["max_duration"].as<int>(300);
            
            auto video = recording["video"];
            config_.codec = video["codec"].as<std::string>("x264");
            config_.preset = video["preset"].as<std::string>("ultrafast");
            config_.crf = video["crf"].as<int>(23);
            config_.width = video["width"].as<int>(1280);
            config_.height = video["height"].as<int>(720);
            config_.framerate = video["framerate"].as<int>(30);
            config_.bitrate = video["bitrate"].as<std::string>("2M");
            
            // Load audio configuration
            auto audio = recording["audio"];
            config_.audio_enabled = audio["enabled"].as<bool>(false);
            config_.audio_device = audio["device"].as<std::string>("hw:0,0");
            config_.audio_sample_rate = audio["sample_rate"].as<int>(48000);
            config_.audio_channels = audio["channels"].as<int>(1);
            config_.audio_codec = audio["codec"].as<std::string>("mp3");
            config_.audio_bitrate = audio["bitrate"].as<int>(128);
            
            // Create output directory
            std::filesystem::create_directories(config_.output_dir);
            
            return createPipeline();
        } catch (const std::exception& e) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool createPipeline() {
        // Initialize GStreamer
        gst_init(nullptr, nullptr);
        
        // Generate filename
        current_filename_ = generateFilename();
        std::string full_path = config_.output_dir + current_filename_;
        
        // Build GStreamer pipeline with video and optional audio
        std::string pipeline_str;
        
        if (config_.audio_enabled) {
            // Video + Audio pipeline with proper sync
            pipeline_str = "appsrc name=src caps=video/x-raw,format=BGR,width=" + 
                          std::to_string(config_.width) + ",height=" + 
                          std::to_string(config_.height) + ",framerate=" + 
                          std::to_string(config_.framerate) + "/1" +
                          " ! videoconvert ! queue ! x264enc bitrate=2000 ! mux. " +
                          "alsasrc device=" + config_.audio_device + 
                          " ! audioconvert ! audioresample ! audio/x-raw,rate=" + 
                          std::to_string(config_.audio_sample_rate) + ",channels=" + 
                          std::to_string(config_.audio_channels) + 
                          " ! queue ! lamemp3enc bitrate=" + std::to_string(config_.audio_bitrate) + 
                          " ! mux. avimux name=mux ! filesink location=" + full_path;
        } else {
            // Video only pipeline
            pipeline_str = "appsrc name=src caps=video/x-raw,format=BGR,width=" + 
                          std::to_string(config_.width) + ",height=" + 
                          std::to_string(config_.height) + ",framerate=" + 
                          std::to_string(config_.framerate) + "/1" +
                          " ! videoconvert ! x264enc bitrate=2000" + 
                          " ! avimux ! filesink location=" + full_path;
        }
        
        std::cout << "Recording pipeline: " << pipeline_str << std::endl;
        
        // Create pipeline
        pipeline_ = gst_parse_launch(pipeline_str.c_str(), nullptr);
        if (!pipeline_) {
            std::cerr << "Failed to create recording pipeline" << std::endl;
            return false;
        }
        
        // Get appsrc element
        appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
        if (!appsrc_) {
            std::cerr << "Failed to get appsrc element" << std::endl;
            return false;
        }
        
        // Configure appsrc
        g_object_set(appsrc_, "format", GST_FORMAT_TIME, nullptr);
        g_object_set(appsrc_, "is-live", TRUE, nullptr);
        
        initialized_ = true;
        return true;
    }
    
    bool start() {
        if (!initialized_) {
            std::cerr << "Recorder not initialized" << std::endl;
            return false;
        }
        
        if (!config_.enabled) {
            std::cout << "Recording disabled in config" << std::endl;
            return true;
        }
        
        recording_ = true;
        recording_start_time_ = std::chrono::steady_clock::now();
        
        // Start pipeline
        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start recording pipeline" << std::endl;
            return false;
        }
        
        // Start recording thread
        recording_ = true;
        recording_start_time_ = std::chrono::steady_clock::now();
        pipeline_start_time_ = gst_clock_get_time(gst_element_get_clock(pipeline_));
        recording_thread_ = std::thread(&GStreamerRecorder::recordingLoop, this);
        
        std::cout << "Recording started: " << current_filename_ << std::endl;
        return true;
    }
    
    void stop() {
        recording_ = false;
        
        // Send End of Stream signal to appsrc
        if (appsrc_) {
            gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
        }
        
        // Wait for recording thread to finish processing
        if (recording_thread_.joinable()) {
            recording_thread_.join();
        }
        
        // Wait longer for EOS to be processed
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Stop pipeline
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }
        
        // Clear frame queue
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!frame_queue_.empty()) {
            frame_queue_.pop();
        }
    }
    
    void addFrame(const cv::Mat& frame) {
        if (!recording_ || !config_.enabled) {
            return;
        }
        
        // Check if we should record this frame
        bool should_record = false;
        if (config_.continuous) {
            should_record = true;
        } else if (config_.event_based && event_detected_) {
            should_record = true;
        }
        
        if (should_record) {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (frame_queue_.size() < 10) { // Limit queue size
                frame_queue_.push(frame.clone());
            }
            queue_condition_.notify_one();
        }
    }
    
    void triggerEvent() {
        if (config_.event_based) {
            event_detected_ = true;
            last_event_time_ = std::chrono::steady_clock::now();
            std::cout << "Event detected - starting recording" << std::endl;
        }
    }
    
    void clearEvent() {
        event_detected_ = false;
    }
    
    // Performance metrics
    uint64_t getFramesRecorded() const { return frames_recorded_.load(); }
    uint64_t getFilesCreated() const { return files_created_.load(); }
    bool isRecording() const { return recording_.load(); }
    
private:
    void recordingLoop() {
        while (recording_) {
            cv::Mat frame;
            
            // Get frame from queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (queue_condition_.wait_for(lock, std::chrono::milliseconds(100),
                                            [this] { return !frame_queue_.empty() || !recording_; })) {
                    if (!frame_queue_.empty()) {
                        frame = frame_queue_.front();
                        frame_queue_.pop();
                    }
                }
            }
            
            if (frame.empty()) {
                continue;
            }
            
            // Send frame to GStreamer
            sendFrame(frame);
            frames_recorded_++;
            
            // Check if we need to create a new file
            if (shouldCreateNewFile()) {
                createNewFile();
            }
        }
    }
    
    void sendFrame(const cv::Mat& frame) {
        // Convert OpenCV Mat to GStreamer buffer
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, frame.total() * frame.elemSize(), nullptr);
        GstMapInfo map;
        
        if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            memcpy(map.data, frame.data, frame.total() * frame.elemSize());
            gst_buffer_unmap(buffer, &map);
            
            // Set proper timestamp with duration
            GstClockTime timestamp = pipeline_start_time_ + (frames_recorded_.load() * GST_SECOND / config_.framerate);
            GST_BUFFER_PTS(buffer) = timestamp;
            GST_BUFFER_DTS(buffer) = timestamp;
            GST_BUFFER_DURATION(buffer) = GST_SECOND / config_.framerate;
            
            // Push buffer to appsrc
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
            if (ret != GST_FLOW_OK) {
                std::cerr << "Failed to push frame to pipeline: " << ret << std::endl;
                gst_buffer_unref(buffer);
            }
        } else {
            std::cerr << "Failed to map buffer" << std::endl;
            gst_buffer_unref(buffer);
        }
    }
    
    bool shouldCreateNewFile() {
        auto now = std::chrono::steady_clock::now();
        
        // Check duration
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - recording_start_time_).count();
        if (duration >= config_.max_duration) {
            std::cout << "Duration limit reached: " << duration << " seconds" << std::endl;
            return true;
        }
        
        // Check actual file size
        std::string full_path = config_.output_dir + current_filename_;
        std::ifstream file(full_path, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            auto file_size_mb = file.tellg() / (1024 * 1024); // Convert to MB
            if (file_size_mb >= config_.max_file_size) {
                std::cout << "File size limit reached: " << file_size_mb << " MB" << std::endl;
                return true;
            }
        }
        
        return false;
    }
    
    void createNewFile() {
        // Stop current pipeline
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        
        // Generate new filename
        current_filename_ = generateFilename();
        std::string full_path = config_.output_dir + current_filename_;
        
        // Update filesink location
        g_object_set(filesink_, "location", full_path.c_str(), nullptr);
        
        // Restart pipeline
        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to restart recording pipeline" << std::endl;
        } else {
            files_created_++;
            recording_start_time_ = std::chrono::steady_clock::now();
            std::cout << "Created new recording file: " << current_filename_ << std::endl;
        }
    }
    
    std::string generateFilename() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);
        
        char buffer[256];
        strftime(buffer, sizeof(buffer), config_.filename_template.c_str(), &tm);
        return std::string(buffer);
    }
};

// Example usage and testing
int main(int argc, char* argv[]) {
    std::string config_file = "configs/pipeline.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    GStreamerRecorder recorder;
    
    if (!recorder.initialize(config_file)) {
        std::cerr << "Failed to initialize recorder" << std::endl;
        return -1;
    }
    
    if (!recorder.start()) {
        std::cerr << "Failed to start recorder" << std::endl;
        return -1;
    }
    
    // Simulate recording for 30 seconds
    std::cout << "Recording test frames for 30 seconds..." << std::endl;
    
    auto start_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    
    while (std::chrono::duration_cast<std::chrono::seconds>(
           std::chrono::steady_clock::now() - start_time).count() < 30) {
        
        // Create test frame
        cv::Mat test_frame = cv::Mat::zeros(720, 1280, CV_8UC3);
        cv::putText(test_frame, "Test Frame " + std::to_string(frame_count++),
                   cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 2.0,
                   cv::Scalar(255, 255, 255), 2);
        
        // Add frame to recorder
        recorder.addFrame(test_frame);
        
        // Simulate event detection every 5 seconds
        if (frame_count % 150 == 0) {
            recorder.triggerEvent();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }
    
    recorder.stop();
    
    std::cout << "Recording completed:" << std::endl;
    std::cout << "Frames recorded: " << recorder.getFramesRecorded() << std::endl;
    std::cout << "Files created: " << recorder.getFilesCreated() << std::endl;
    
    return 0;
}
