// Stub YAML implementation for when yaml-cpp is not available
#ifndef STUB_YAML_H
#define STUB_YAML_H

#include <string>
#include <vector>
#include <map>
#include <iterator>

namespace YAML {
    class Node {
    private:
        std::map<std::string, Node> map_data_;
        std::string string_data_;
        int int_data_ = 0;
        float float_data_ = 0.0f;
        bool bool_data_ = false;
        std::vector<Node> sequence_data_;
        bool is_map_ = false;
        bool is_string_ = false;
        bool is_int_ = false;
        bool is_float_ = false;
        bool is_bool_ = false;
        bool is_sequence_ = false;

    public:
        Node() = default;
        Node(const std::string& value) : string_data_(value), is_string_(true) {}
        Node(int value) : int_data_(value), is_int_(true) {}
        Node(float value) : float_data_(value), is_float_(true) {}
        Node(bool value) : bool_data_(value), is_bool_(true) {}

        Node& operator[](const std::string& key) {
            is_map_ = true;
            return map_data_[key];
        }

        Node& operator[](const char* key) {
            is_map_ = true;
            return map_data_[std::string(key)];
        }

        const Node& operator[](const std::string& key) const {
            auto it = map_data_.find(key);
            if (it != map_data_.end()) {
                return it->second;
            }
            static Node empty;
            return empty;
        }

        const Node& operator[](const char* key) const {
            auto it = map_data_.find(std::string(key));
            if (it != map_data_.end()) {
                return it->second;
            }
            static Node empty;
            return empty;
        }

        template<typename T>
        T as(const T& default_value = T{}) const {
            if constexpr (std::is_same_v<T, std::string>) {
                return is_string_ ? string_data_ : default_value;
            } else if constexpr (std::is_same_v<T, int>) {
                return is_int_ ? int_data_ : default_value;
            } else if constexpr (std::is_same_v<T, float>) {
                return is_float_ ? float_data_ : default_value;
            } else if constexpr (std::is_same_v<T, bool>) {
                return is_bool_ ? bool_data_ : default_value;
            }
            return default_value;
        }

        // Specialized versions for common types
        std::string as(const std::string& default_value = "") const {
            return is_string_ ? string_data_ : default_value;
        }

        int as(int default_value = 0) const {
            return is_int_ ? int_data_ : default_value;
        }

        float as(float default_value = 0.0f) const {
            return is_float_ ? float_data_ : default_value;
        }

        bool as(bool default_value = false) const {
            return is_bool_ ? bool_data_ : default_value;
        }


        bool IsSequence() const { return is_sequence_; }
        bool IsMap() const { return is_map_; }
        bool IsScalar() const { return is_string_ || is_int_ || is_float_ || is_bool_; }

        size_t size() const {
            if (is_sequence_) return sequence_data_.size();
            if (is_map_) return map_data_.size();
            return 0;
        }

        const Node& operator[](size_t index) const {
            if (is_sequence_ && index < sequence_data_.size()) {
                return sequence_data_[index];
            }
            static Node empty;
            return empty;
        }

        void push_back(const Node& node) {
            is_sequence_ = true;
            sequence_data_.push_back(node);
        }

        // Iterator support for range-based for loops
        auto begin() const {
            if (is_sequence_) {
                return sequence_data_.begin();
            }
            return sequence_data_.end();
        }

        auto end() const {
            return sequence_data_.end();
        }

        // Boolean conversion for if statements
        operator bool() const {
            return is_map_ || is_string_ || is_int_ || is_float_ || is_bool_ || is_sequence_;
        }
    };

    inline Node LoadFile(const std::string& filename) {
        // Return a node with default configuration
        Node config;
        
        // Camera configuration
        Node camera;
        camera["device"] = Node("/dev/video0");
        camera["width"] = Node(640);
        camera["height"] = Node(480);
        camera["framerate"] = Node(15);
        camera["format"] = Node("YUYV");  // Force YUYV format
        camera["dmabuf"] = Node(true);
        config["camera"] = camera;
        
        // Pipeline configuration
        Node pipeline;
        Node recording;
        recording["enabled"] = Node(true);
        recording["continuous"] = Node(true);  // Enable continuous recording
        recording["event_based"] = Node(false);
        
        Node video;
        video["codec"] = Node("x264");
        video["preset"] = Node("ultrafast");
        video["crf"] = Node(23);
        video["width"] = Node(1280);
        video["height"] = Node(720);
        video["framerate"] = Node(30);
        video["bitrate"] = Node("2M");
        recording["video"] = video;
        
        recording["output_dir"] = Node("recordings/");
        recording["filename_template"] = Node("plowpilot_%Y%m%d_%H%M%S.avi");
        recording["max_file_size"] = Node(100);
        recording["max_duration"] = Node(300);
        
        Node audio;
        audio["enabled"] = Node(true);
        audio["device"] = Node("hw:0,0");
        audio["sample_rate"] = Node(48000);
        audio["channels"] = Node(1);
        audio["codec"] = Node("mp3");
        audio["bitrate"] = Node(128);
        recording["audio"] = audio;
        
        pipeline["recording"] = recording;
        config["pipeline"] = pipeline;
        config["recording"] = recording;  // Also add at root level for compatibility
        
        return config;
    }
}

#endif // STUB_YAML_H
