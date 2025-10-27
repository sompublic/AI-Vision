cmake_minimum_required(VERSION 3.16)
project(PlowPilot-AI-Vision VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA is not needed for this project - we only use TensorRT
message(STATUS "Building without CUDA support - using TensorRT only")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found - using alternative YAML parsing")
    add_definitions(-DUSE_ALTERNATIVE_YAML)
endif()

# Find GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)

# Find TensorRT
set(TENSORRT_ROOT "/usr/src/tensorrt")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/include/aarch64-linux-gnu
    PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/lib/aarch64-linux-gnu
    PATH_SUFFIXES lib lib64 lib/x64)

# Find CUDA headers for TensorRT
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
    HINTS /usr/local/cuda-12.6/targets/aarch64-linux/include
    PATH_SUFFIXES include)

# Find MQTT (optional)
find_package(PahoMqttCpp QUIET)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link directories
link_directories(${GSTREAMER_LIBRARY_DIRS})

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# No CUDA flags needed

# Define preprocessor macros
add_definitions(-DUSE_TENSORRT)
if(PahoMqttCpp_FOUND)
    add_definitions(-DUSE_MQTT)
endif()

# Source files for main executable (without main functions)
set(MAIN_SOURCES
    src/main.cpp
)

# Create main executable
add_executable(plowpilot ${MAIN_SOURCES})

# Link libraries
target_link_libraries(plowpilot
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    pthread
    dl
)
if(TENSORRT_LIBRARY)
    target_link_libraries(plowpilot ${TENSORRT_LIBRARY})
endif()
if(yaml-cpp_FOUND)
    target_link_libraries(plowpilot yaml-cpp)
endif()

# Add MQTT if available
if(PahoMqttCpp_FOUND)
    target_link_libraries(plowpilot PahoMqttCpp::paho-mqttpp3)
endif()

# Set RPATH for runtime library discovery
set_target_properties(plowpilot PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)

# Create individual executables for testing
add_executable(test_capture src/capture_gst.cpp)
target_link_libraries(test_capture
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    pthread
    dl
)
if(yaml-cpp_FOUND)
    target_link_libraries(test_capture yaml-cpp)
endif()

# Test inference only if TensorRT is available
if(TENSORRT_LIBRARY)
    add_executable(test_inference src/infer_trt.cpp)
    target_link_libraries(test_inference
        ${OpenCV_LIBS}
        ${TENSORRT_LIBRARY}
        pthread
        dl
    )
    if(yaml-cpp_FOUND)
        target_link_libraries(test_inference yaml-cpp)
    endif()
endif()

add_executable(test_pipeline src/pipeline.cpp)
target_link_libraries(test_pipeline
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    pthread
    dl
)
if(TENSORRT_LIBRARY)
    target_link_libraries(test_pipeline ${TENSORRT_LIBRARY})
endif()
if(yaml-cpp_FOUND)
    target_link_libraries(test_pipeline yaml-cpp)
endif()

add_executable(test_recording src/record_gst.cpp)
target_link_libraries(test_recording
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    pthread
    dl
)
if(yaml-cpp_FOUND)
    target_link_libraries(test_recording yaml-cpp)
endif()

add_executable(test_events src/events_mqtt.cpp)
target_link_libraries(test_events
    ${OpenCV_LIBS}
    pthread
    dl
)
if(yaml-cpp_FOUND)
    target_link_libraries(test_events yaml-cpp)
endif()

if(PahoMqttCpp_FOUND)
    target_link_libraries(test_events PahoMqttCpp::paho-mqttpp3)
endif()

# Install targets
install(TARGETS plowpilot test_capture test_inference test_pipeline test_recording test_events
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install configuration files
install(DIRECTORY configs/
    DESTINATION etc/plowpilot
    FILES_MATCHING PATTERN "*.yaml"
)

# Install systemd service
install(FILES systemd/plowpilot.service
    DESTINATION /etc/systemd/system
)

# Create data directories
install(DIRECTORY DESTINATION var/lib/plowpilot/data)
install(DIRECTORY DESTINATION var/lib/plowpilot/models)
install(DIRECTORY DESTINATION var/lib/plowpilot/recordings)
install(DIRECTORY DESTINATION var/lib/plowpilot/logs)

# CPack configuration for creating packages
set(CPACK_PACKAGE_NAME "plowpilot-ai-vision")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PlowPilot AI-Vision: Real-time video analytics on NVIDIA Jetson")
set(CPACK_PACKAGE_VENDOR "PlowPilot Team")

include(CPack)

# Print configuration summary
message(STATUS "PlowPilot AI-Vision Configuration Summary:")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  GStreamer: ${GSTREAMER_VERSION}")
message(STATUS "  TensorRT: ${TENSORRT_LIBRARY}")
message(STATUS "  CUDA: ${CUDA_VERSION}")
message(STATUS "  MQTT: ${PahoMqttCpp_FOUND}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
