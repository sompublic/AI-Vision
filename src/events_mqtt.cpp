/**
 * @file events_mqtt.cpp
 * @brief Event detection and MQTT alerts for PlowPilot AI-Vision
 * @author PlowPilot Team
 * @date 2025-01-25
 * 
 * P3: Event triggers and MQTT alerts
 * Target: Alert latency ≤500 ms from event occurrence
 * 24h soak: zero crashes, no unrecovered camera disconnect
 */

#include <mqtt/async_client.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

struct Event {
    std::string type;
    std::string class_name;
    float confidence;
    float x1, y1, x2, y2;
    std::chrono::steady_clock::time_point timestamp;
    uint64_t frame_id;
    
    Event(const std::string& t, const std::string& cn, float conf, 
          float x1_, float y1_, float x2_, float y2_, uint64_t fid) 
        : type(t), class_name(cn), confidence(conf),
          x1(x1_), y1(y1_), x2(x2_), y2(y2_), frame_id(fid) {
        timestamp = std::chrono::steady_clock::now();
    }
};

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    std::string class_name;
};

class MQTTClient {
private:
    mqtt::async_client* client_;
    mqtt::connect_options conn_opts_;
    
    std::string broker_host_;
    int broker_port_;
    std::string topic_prefix_;
    int qos_;
    bool retain_;
    
    std::atomic<bool> connected_;
    std::atomic<bool> initialized_;
    
public:
    MQTTClient() : client_(nullptr), broker_port_(1883), qos_(1), 
                   retain_(false), connected_(false), initialized_(false) {}
    
    ~MQTTClient() {
        disconnect();
        if (client_) {
            delete client_;
        }
    }
    
    bool initialize(const std::string& config_file) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            auto mqtt = config["mqtt"];
            
            if (!mqtt["enabled"].as<bool>(false)) {
                std::cout << "MQTT disabled in config" << std::endl;
                return true;
            }
            
            broker_host_ = mqtt["broker_host"].as<std::string>("localhost");
            broker_port_ = mqtt["broker_port"].as<int>(1883);
            topic_prefix_ = mqtt["topic_prefix"].as<std::string>("plowpilot/");
            qos_ = mqtt["qos"].as<int>(1);
            retain_ = mqtt["retain"].as<bool>(false);
            
            // Create MQTT client
            std::string client_id = "plowpilot_" + std::to_string(getpid());
            client_ = new mqtt::async_client("tcp://" + broker_host_ + ":" + std::to_string(broker_port_), client_id);
            
            // Set connection options
            conn_opts_.set_keep_alive_interval(20);
            conn_opts_.set_clean_session(true);
            
            // Set callbacks
            client_->set_connection_lost_handler([this](const std::string& cause) {
                std::cout << "MQTT connection lost: " << cause << std::endl;
                connected_ = false;
            });
            
            client_->set_connected_handler([this](const std::string& cause) {
                std::cout << "MQTT connected: " << cause << std::endl;
                connected_ = true;
            });
            
            initialized_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading MQTT config: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool connect() {
        if (!initialized_ || !client_) {
            return false;
        }
        
        try {
            client_->connect(conn_opts_)->wait();
            return true;
        } catch (const mqtt::exception& e) {
            std::cerr << "MQTT connection failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    void disconnect() {
        if (client_ && connected_) {
            try {
                client_->disconnect()->wait();
            } catch (const mqtt::exception& e) {
                std::cerr << "MQTT disconnect failed: " << e.what() << std::endl;
            }
        }
        connected_ = false;
    }
    
    bool publish(const std::string& topic, const std::string& message) {
        if (!connected_ || !client_) {
            return false;
        }
        
        try {
            std::string full_topic = topic_prefix_ + topic;
            mqtt::message_ptr msg = mqtt::make_message(full_topic, message);
            msg->set_qos(qos_);
            msg->set_retained(retain_);
            
            client_->publish(msg)->wait();
            return true;
        } catch (const mqtt::exception& e) {
            std::cerr << "MQTT publish failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool isConnected() const { return connected_.load(); }
};

class EventDetector {
private:
    // Event detection configuration
    std::vector<std::string> event_types_;
    float confidence_threshold_;
    float nms_threshold_;
    int min_area_, max_area_;
    
    // Event persistence
    float min_duration_, max_duration_, cooldown_;
    std::chrono::steady_clock::time_point last_event_time_;
    std::atomic<bool> event_active_;
    
    // Event queue
    std::queue<Event> event_queue_;
    std::mutex event_mutex_;
    std::condition_variable event_condition_;
    
    // MQTT client
    std::unique_ptr<MQTTClient> mqtt_client_;
    
    // Performance monitoring
    std::atomic<uint64_t> events_detected_;
    std::atomic<uint64_t> events_published_;
    std::chrono::steady_clock::time_point start_time_;
    
public:
    EventDetector() : confidence_threshold_(0.5), nms_threshold_(0.45),
                      min_area_(100), max_area_(100000),
                      min_duration_(1.0), max_duration_(60.0), cooldown_(5.0),
                      event_active_(false), events_detected_(0), events_published_(0) {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    bool initialize(const std::string& config_file) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            auto events = config["events"];
            
            // Load event configuration
            confidence_threshold_ = events["detection"]["confidence_threshold"].as<float>(0.5);
            nms_threshold_ = events["detection"]["nms_threshold"].as<float>(0.45);
            min_area_ = events["detection"]["min_area"].as<int>(100);
            max_area_ = events["detection"]["max_area"].as<int>(100000);
            
            // Load event types
            if (events["event_types"]) {
                for (const auto& event_type : events["event_types"]) {
                    event_types_.push_back(event_type.as<std::string>());
                }
            }
            
            // Load persistence settings
            auto persistence = events["persistence"];
            min_duration_ = persistence["min_duration"].as<float>(1.0);
            max_duration_ = persistence["max_duration"].as<float>(60.0);
            cooldown_ = persistence["cooldown"].as<float>(5.0);
            
            // Initialize MQTT client
            mqtt_client_ = std::make_unique<MQTTClient>();
            if (!mqtt_client_->initialize(config_file)) {
                std::cerr << "Failed to initialize MQTT client" << std::endl;
                return false;
            }
            
            if (!mqtt_client_->connect()) {
                std::cerr << "Failed to connect to MQTT broker" << std::endl;
                // Continue without MQTT - it's optional
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading event config: " << e.what() << std::endl;
            return false;
        }
    }
    
    void processDetections(const std::vector<Detection>& detections, uint64_t frame_id) {
        for (const auto& detection : detections) {
            // Check if this detection matches our event types
            if (std::find(event_types_.begin(), event_types_.end(), detection.class_name) != event_types_.end()) {
                // Check confidence threshold
                if (detection.confidence >= confidence_threshold_) {
                    // Check area constraints
                    float area = (detection.x2 - detection.x1) * (detection.y2 - detection.y1);
                    if (area >= min_area_ && area <= max_area_) {
                        // Create event
                        Event event("detection", detection.class_name, detection.confidence,
                                  detection.x1, detection.y1, detection.x2, detection.y2, frame_id);
                        
                        // Check cooldown
                        auto now = std::chrono::steady_clock::now();
                        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_event_time_).count() >= cooldown_) {
                            addEvent(event);
                            last_event_time_ = now;
                        }
                    }
                }
            }
        }
    }
    
    void addEvent(const Event& event) {
        std::lock_guard<std::mutex> lock(event_mutex_);
        event_queue_.push(event);
        events_detected_++;
        event_condition_.notify_one();
    }
    
    bool getEvent(Event& event, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(event_mutex_);
        if (event_condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                    [this] { return !event_queue_.empty(); })) {
            if (!event_queue_.empty()) {
                event = event_queue_.front();
                event_queue_.pop();
                return true;
            }
        }
        return false;
    }
    
    void publishEvent(const Event& event) {
        if (!mqtt_client_ || !mqtt_client_->isConnected()) {
            return;
        }
        
        // Create JSON message
        std::string message = "{";
        message += "\"type\":\"" + event.type + "\",";
        message += "\"class\":\"" + event.class_name + "\",";
        message += "\"confidence\":" + std::to_string(event.confidence) + ",";
        message += "\"bbox\":[" + std::to_string(event.x1) + "," + std::to_string(event.y1) + 
                  "," + std::to_string(event.x2) + "," + std::to_string(event.y2) + "],";
        message += "\"frame_id\":" + std::to_string(event.frame_id) + ",";
        message += "\"timestamp\":" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                      event.timestamp.time_since_epoch()).count());
        message += "}";
        
        // Publish to MQTT
        std::string topic = "events/" + event.type;
        if (mqtt_client_->publish(topic, message)) {
            events_published_++;
            std::cout << "Published event: " << event.class_name 
                      << " (confidence: " << event.confidence << ")" << std::endl;
        }
    }
    
    // Performance metrics
    uint64_t getEventsDetected() const { return events_detected_.load(); }
    uint64_t getEventsPublished() const { return events_published_.load(); }
    bool isEventActive() const { return event_active_.load(); }
    
    void setEventActive(bool active) { event_active_ = active; }
};

// Example usage and testing
int main(int argc, char* argv[]) {
    std::string config_file = "configs/pipeline.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    EventDetector detector;
    
    if (!detector.initialize(config_file)) {
        std::cerr << "Failed to initialize event detector" << std::endl;
        return -1;
    }
    
    std::cout << "Event detector initialized" << std::endl;
    
    // Simulate detections
    std::vector<Detection> test_detections = {
        {100, 100, 200, 200, 0.8f, 0, "person"},
        {300, 300, 400, 400, 0.7f, 1, "car"},
        {500, 500, 600, 600, 0.6f, 2, "bicycle"}
    };
    
    std::cout << "Processing test detections..." << std::endl;
    
    for (int i = 0; i < 10; i++) {
        detector.processDetections(test_detections, i);
        
        // Check for events
        Event event;
        if (detector.getEvent(event, 100)) {
            std::cout << "Event detected: " << event.class_name 
                      << " (confidence: " << event.confidence << ")" << std::endl;
            detector.publishEvent(event);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "Event detection stats:" << std::endl;
    std::cout << "Events detected: " << detector.getEventsDetected() << std::endl;
    std::cout << "Events published: " << detector.getEventsPublished() << std::endl;
    
    return 0;
}
