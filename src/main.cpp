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
#include <opencv2/opencv.hpp>

// Include pipeline components
#define INCLUDED_IN_MAIN
#define INCLUDED_IN_PIPELINE
#include "capture_gst.cpp"
#include "infer_trt.cpp"
#include "pipeline.cpp"

// Global variables for signal handling
std::atomic<bool> g_running{true};
AsyncPipeline* g_pipeline = nullptr;

void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
    g_running = false;
    if (g_pipeline) {
        g_pipeline->stop();
    }
}

int main(int argc, char* argv[]) {
    std::cout << "PlowPilot AI-Vision - Real-time Video Analytics Pipeline" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
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
        // Create pipeline instance
        auto pipeline = std::make_unique<AsyncPipeline>();
        g_pipeline = pipeline.get();
        
        // Initialize pipeline based on phase
        if (phase == "P1") {
            std::cout << "Running Phase 1: Capture & Profiling" << std::endl;
            
            if (!pipeline->initialize(config_file)) {
                std::cerr << "Failed to initialize pipeline for P1" << std::endl;
                return -1;
            }
            
        } else if (phase == "P2") {
            std::cout << "Running Phase 2: Inference & Async Pipeline" << std::endl;
            
            if (!pipeline->initialize(config_file)) {
                std::cerr << "Failed to initialize pipeline for P2" << std::endl;
                return -1;
            }
            
        } else if (phase == "P3") {
            std::cout << "Running Phase 3: Full Pipeline with Recording & Events" << std::endl;
            
            if (!pipeline->initialize(config_file)) {
                std::cerr << "Failed to initialize pipeline for P3" << std::endl;
                return -1;
            }
            
        } else {
            std::cerr << "Invalid phase: " << phase << std::endl;
            std::cerr << "Valid phases: P1, P2, P3" << std::endl;
            return -1;
        }
        
        // Start pipeline
        std::cout << "Starting pipeline..." << std::endl;
        if (!pipeline->start()) {
            std::cerr << "Failed to start pipeline" << std::endl;
            return -1;
        }
        
        std::cout << "Pipeline running. Press Ctrl+C to stop." << std::endl;
        
        // Main monitoring loop
        auto start_time = std::chrono::steady_clock::now();
        auto last_stats_time = start_time;
        
        while (g_running && pipeline->isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Print stats every 10 seconds
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count() >= 10) {
                std::cout << "[Stats] Runtime: " << duration << "s" << std::endl;
                std::cout << "[Stats] FPS: " << pipeline->getFPS() << std::endl;
                std::cout << "[Stats] Total frames: " << pipeline->getTotalFrames() << std::endl;
                std::cout << "[Stats] Processed frames: " << pipeline->getProcessedFrames() << std::endl;
                std::cout << "[Stats] Dropped frames: " << pipeline->getDroppedFrames() << std::endl;
                std::cout << "---" << std::endl;
                last_stats_time = now;
            }
        }
        
        // Stop pipeline
        std::cout << "\nStopping pipeline..." << std::endl;
        pipeline->stop();
        g_pipeline = nullptr;
        
        // Print final stats
        std::cout << "\nFinal Statistics:" << std::endl;
        std::cout << "  Total frames: " << pipeline->getTotalFrames() << std::endl;
        std::cout << "  Processed frames: " << pipeline->getProcessedFrames() << std::endl;
        std::cout << "  Dropped frames: " << pipeline->getDroppedFrames() << std::endl;
        auto final_duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
        if (final_duration > 0) {
            std::cout << "  Average FPS: " << (pipeline->getTotalFrames() / final_duration) << std::endl;
        }
        
        std::cout << "Shutdown complete" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (g_pipeline) {
            g_pipeline->stop();
        }
        return -1;
    }
}
