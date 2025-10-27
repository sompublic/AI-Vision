// Stub MQTT implementation for when MQTT is not available
#ifndef STUB_MQTT_H
#define STUB_MQTT_H

#include <string>
#include <memory>
#include <functional>

namespace mqtt {
    class exception : public std::exception {
    private:
        std::string message_;
    public:
        exception(const std::string& msg) : message_(msg) {}
        const char* what() const noexcept override { return message_.c_str(); }
    };

    class message {
    private:
        std::string topic_;
        std::string payload_;
        int qos_ = 0;
        bool retain_ = false;

    public:
        message(const std::string& topic, const std::string& payload) 
            : topic_(topic), payload_(payload) {}
        
        void set_qos(int qos) { qos_ = qos; }
        void set_retained(bool retain) { retain_ = retain; }
        
        const std::string& get_topic() const { return topic_; }
        const std::string& get_payload() const { return payload_; }
        int get_qos() const { return qos_; }
        bool is_retained() const { return retain_; }
    };

    using message_ptr = std::shared_ptr<message>;

    inline message_ptr make_message(const std::string& topic, const std::string& payload) {
        return std::make_shared<message>(topic, payload);
    }

    class connect_options {
    private:
        int keep_alive_interval_ = 60;
        bool clean_session_ = true;

    public:
        void set_keep_alive_interval(int interval) { keep_alive_interval_ = interval; }
        void set_clean_session(bool clean) { clean_session_ = clean; }
        int get_keep_alive_interval() const { return keep_alive_interval_; }
        bool get_clean_session() const { return clean_session_; }
    };

    class delivery_token {
    public:
        delivery_token* operator->() { return this; }
        void wait() const {
            // Stub implementation - do nothing
        }
    };

    class async_client {
    private:
        std::string server_uri_;
        std::string client_id_;
        bool connected_ = false;

    public:
        async_client(const std::string& server_uri, const std::string& client_id)
            : server_uri_(server_uri), client_id_(client_id) {}

        delivery_token connect(const connect_options& opts) {
            connected_ = true;
            return delivery_token{};
        }

        delivery_token disconnect() {
            connected_ = false;
            return delivery_token{};
        }

        delivery_token publish(message_ptr msg) {
            // Stub implementation - just return a token
            return delivery_token{};
        }

        void set_connection_lost_handler(std::function<void(const std::string&)> handler) {
            // Stub implementation - do nothing
        }

        void set_connected_handler(std::function<void(const std::string&)> handler) {
            // Stub implementation - do nothing
        }

        bool is_connected() const { return connected_; }
    };
}

#endif // STUB_MQTT_H
