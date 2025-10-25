cmake_minimum_required(VERSION 3.16)
project(PlowPilot-AI-Vision VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(yaml-cpp REQUIRED)

# Find GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY NvInfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

# Find CUDA
find_package(CUDA REQUIRED)

# Find MQTT (optional)
find_package(PahoMqttCpp QUIET)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link directories
link_directories(${GSTREAMER_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

# Define preprocessor macros
add_definitions(-DUSE_TENSORRT)
if(PahoMqttCpp_FOUND)
    add_definitions(-DUSE_MQTT)
endif()

# Source files
set(SOURCES
    src/capture_gst.cpp
    src/infer_trt.cpp
    src/pipeline.cpp
    src/record_gst.cpp
    src/events_mqtt.cpp
)

# Create main executable
add_executable(plowpilot ${SOURCES})

# Link libraries
target_link_libraries(plowpilot
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
    yaml-cpp
    pthread
    dl
)

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
    yaml-cpp
    pthread
    dl
)

add_executable(test_inference src/infer_trt.cpp)
target_link_libraries(test_inference
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
    yaml-cpp
    pthread
    dl
)

add_executable(test_pipeline src/pipeline.cpp)
target_link_libraries(test_pipeline
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
    yaml-cpp
    pthread
    dl
)

add_executable(test_recording src/record_gst.cpp)
target_link_libraries(test_recording
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    yaml-cpp
    pthread
    dl
)

add_executable(test_events src/events_mqtt.cpp)
target_link_libraries(test_events
    ${OpenCV_LIBS}
    yaml-cpp
    pthread
    dl
)

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
