/**
 * @file main.cpp
 * @brief Main entry point for PlowPilot AI-Vision
 * @author PlowPilot Team
 * @date 2025-01-25
 * 
 * Main application entry point that integrates all components
 */

#include <iostream>
#include <memory>
#include <signal.h>
#include <chrono>
#include <thread>
#include <atomic>

// Forward declarations
class GStreamerCapture;
class TensorRTInference;
class GStreamerRecorder;
class EventDetector;

// Global variables for signal handling
std::atomic<bool> g_running{true};

void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::cout << "PlowPilot AI-Vision - Real-time Video Analytics Pipeline" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    // Set up signal handling
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Parse command line arguments
    std::string config_file = "configs/pipeline.yaml";
    std::string phase = "P3"; // Default to full pipeline
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--phase" && i + 1 < argc) {
            phase = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --config <file>  Configuration file (default: configs/pipeline.yaml)" << std::endl;
            std::cout << "  --phase <phase>   Pipeline phase: P1, P2, P3 (default: P3)" << std::endl;
            std::cout << "  --help           Show this help message" << std::endl;
            return 0;
        }
    }
    
    std::cout << "Configuration file: " << config_file << std::endl;
    std::cout << "Pipeline phase: " << phase << std::endl;
    
    try {
        // Initialize components based on phase
        if (phase == "P1") {
            std::cout << "Running Phase 1: Capture & Profiling" << std::endl;
            // TODO: Implement P1-only mode
            std::cout << "P1 mode not yet implemented" << std::endl;
        } else if (phase == "P2") {
            std::cout << "Running Phase 2: Inference & Async Pipeline" << std::endl;
            // TODO: Implement P2 mode
            std::cout << "P2 mode not yet implemented" << std::endl;
        } else if (phase == "P3") {
            std::cout << "Running Phase 3: Full Pipeline with Recording & Events" << std::endl;
            // TODO: Implement P3 mode
            std::cout << "P3 mode not yet implemented" << std::endl;
        } else {
            std::cerr << "Invalid phase: " << phase << std::endl;
            std::cerr << "Valid phases: P1, P2, P3" << std::endl;
            return -1;
        }
        
        // Main loop
        std::cout << "Starting main loop..." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        
        while (g_running) {
            // TODO: Implement main pipeline logic
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Print status every 10 seconds
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            if (duration > 0 && duration % 10 == 0) {
                std::cout << "Running for " << duration << " seconds..." << std::endl;
            }
        }
        
        std::cout << "Shutdown complete" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
