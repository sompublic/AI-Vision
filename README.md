# PlowPilot AI-Vision

**Real-time video analytics pipeline on NVIDIA Jetson Orin Nano (8GB) with Logitech USB webcam**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: NVIDIA Jetson](https://img.shields.io/badge/Platform-NVIDIA%20Jetson-green.svg)](https://developer.nvidia.com/embedded/jetson-orin-nano)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.6.1-red.svg)](https://developer.nvidia.com/tensorrt)
[![GStreamer](https://img.shields.io/badge/GStreamer-1.0-blue.svg)](https://gstreamer.freedesktop.org/)

## Overview

PlowPilot AI-Vision is a high-performance, real-time video analytics system designed for NVIDIA Jetson Orin Nano. It implements a 3-phase iterative development approach for building a complete computer vision pipeline optimized for edge deployment.

### Key Features

- **Real-time Object Detection**: YOLOv8n with TensorRT optimization
- **Low-latency Capture**: GStreamer-based camera pipeline with DMABUF support
- **Async Processing**: Multi-threaded pipeline with bounded queues
- **Event Recording**: Software H.264 encoding with event triggers
- **MQTT Integration**: Real-time alerts and notifications
- **Docker Support**: Containerized deployment with multi-stage builds
- **Systemd Integration**: Production-ready service management
- **Web Interface**: Real-time monitoring and control dashboard

## Hardware Requirements

- **Device**: NVIDIA Jetson Orin Nano Super Dev Kit (8GB)
- **Camera**: Logitech USB webcam (C920/C922-class, 1080p30-capable)
- **Storage**: 128-256GB U3/UHS-I microSD or SSD
- **Cooling**: USB desk fan/ducting (recommended for continuous operation)

## Software Requirements

- **JetPack**: 5.1.2 or later
- **CUDA**: 11.4 or later
- **TensorRT**: 8.6.1 or later
- **GStreamer**: 1.0 or later
- **OpenCV**: 4.5 or later
- **CMake**: 3.16 or later
- **Python**: 3.8 or later

## Project Structure

```
plowpilot/
├── configs/                 # Configuration files
│   ├── camera.yaml         # Camera settings and GStreamer pipelines
│   ├── model.yaml          # Model configuration and TensorRT settings
│   └── pipeline.yaml       # Pipeline configuration and performance targets
├── src/                    # C++ source code
│   ├── capture_gst.cpp     # P1: GStreamer capture with DMABUF
│   ├── infer_trt.cpp       # P2: TensorRT inference wrapper
│   ├── pipeline.cpp        # P2: Async multi-threaded pipeline
│   ├── record_gst.cpp      # P3: GStreamer recording with x264
│   └── events_mqtt.cpp     # P3: Event detection and MQTT alerts
├── scripts/                # Build and utility scripts
│   ├── build_engines.sh    # TensorRT engine builder
│   ├── run_jetson_clocks.sh # Jetson performance tuning
│   ├── bench_tegrastats.sh # Telemetry collection and analysis
│   ├── export_model.py     # YOLOv8n to ONNX export
│   └── web_interface.py    # Web dashboard
├── docker/                 # Containerization
│   ├── Dockerfile         # Multi-stage build (builder + runtime)
│   ├── docker-compose.yml # Development and production deployment
│   └── mosquitto.conf     # MQTT broker configuration
├── systemd/               # Service management
│   ├── plowpilot.service  # Main service
│   ├── plowpilot-clocks.service # Clock configuration
│   └── plowpilot-tensorrt.service # TensorRT engine builder
└── data/                  # Runtime data
    ├── models/            # TensorRT engines and ONNX models
    ├── recordings/        # Video recordings
    ├── logs/             # Application logs
    └── benchmarks/        # Performance telemetry
```

## Development Phases

### Phase 1: Capture & Profiling (P1)
**Goal**: Low-latency, GPU-friendly capture with stable FPS

**Acceptance Criteria**:
- 720p @ 30 FPS sustained (≥28 FPS over 10 min)
- End-to-end latency ≤15 ms p50 / ≤25 ms p95
- Capture+display CPU ≤25% of one core
- No memory leaks (ΔRSS < 50 MB over 30 min)

**Implementation**:
- GStreamer V4L2 capture with DMABUF
- OpenCV integration via GStreamer pipeline
- Performance monitoring and telemetry

### Phase 2: Inference & Async Pipeline (P2)
**Goal**: TensorRT inference in async multi-threaded pipeline

**Acceptance Criteria**:
- 720p @ 30 FPS end-to-end (capture→infer→draw)
- P95 latency ≤80 ms
- Model: YOLOv8n/SSD-MNV2 FP16 (≤20-25 ms inference)
- Stable GPU util 60-90%, no queue growth

**Implementation**:
- TensorRT engine optimization (FP16/INT8)
- Async producer/consumer with bounded queues
- Backpressure handling with drop-oldest policy

### Phase 3: Recording & Events (P3)
**Goal**: Annotated recording, event triggers, MQTT alerts

**Acceptance Criteria**:
- Continuous/event recording at 720p @ 20-30 FPS
- Alert latency ≤500 ms from event occurrence
- 24h soak: zero crashes, no camera disconnect

**Implementation**:
- Software H.264 encoding (x264 ultrafast)
- Event detection and persistence
- MQTT integration for alerts
- Systemd service management

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/sompublic/AI-Vision.git
cd AI-Vision
```

### 2. Build TensorRT Engines
```bash
# Build FP16 engine
./scripts/build_engines.sh

# Build both FP16 and INT8 engines
./scripts/build_engines.sh --int8

# Build engines with benchmarking
./scripts/build_engines.sh --int8 --benchmark
```

### 3. Set Jetson Performance Mode
```bash
# Set maximum performance
sudo ./scripts/run_jetson_clocks.sh max

# Set balanced performance
sudo ./scripts/run_jetson_clocks.sh balanced

# Set power saving mode
sudo ./scripts/run_jetson_clocks.sh power
```

### 4. Build Application
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 5. Run Tests
```bash
# Test capture pipeline
./test_capture

# Test inference
./test_inference

# Test full pipeline
./test_pipeline

# Test recording
./test_recording

# Test events
./test_events
```

### 6. Run Main Application
```bash
# Run with default configuration
./plowpilot

# Run with custom configuration
./plowpilot --config /path/to/config.yaml

# Run with specific phase
./plowpilot --phase P1  # Capture only
./plowpilot --phase P2  # Capture + inference
./plowpilot --phase P3  # Full pipeline
```

## Docker Deployment

### Development Environment
```bash
# Build development container
docker-compose -f docker/docker-compose.yml up development

# Build TensorRT engines
docker-compose -f docker/docker-compose.yml up tensorrt-builder
```

### Production Deployment
```bash
# Build and run production stack
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f plowpilot

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## Systemd Service Management

### Install Service
```bash
# Copy service files
sudo cp systemd/*.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable plowpilot
sudo systemctl enable plowpilot-clocks
```

### Service Control
```bash
# Start service
sudo systemctl start plowpilot

# Stop service
sudo systemctl stop plowpilot

# Restart service
sudo systemctl restart plowpilot

# View status
sudo systemctl status plowpilot

# View logs
sudo journalctl -u plowpilot -f
```

## Configuration

### Camera Configuration (`configs/camera.yaml`)
```yaml
camera:
  device: "/dev/video0"
  width: 1280
  height: 720
  framerate: 30
  format: "MJPG"
  gst_pipeline: "v4l2src device=/dev/video0 ! video/x-raw,format=MJPG,width=1280,height=720,framerate=30/1 ! jpegdec ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true"
```

### Model Configuration (`configs/model.yaml`)
```yaml
model:
  name: "yolov8n"
  trt_engine_path: "models/yolov8n_fp16.trt"
  input:
    width: 640
    height: 640
    channels: 3
    batch_size: 1
    data_type: "fp16"
  output:
    confidence_threshold: 0.5
    nms_threshold: 0.45
    max_detections: 100
```

### Pipeline Configuration (`configs/pipeline.yaml`)
```yaml
pipeline:
  queues:
    capture_queue_size: 4
    inference_queue_size: 2
    display_queue_size: 2
    recording_queue_size: 8
    drop_policy: "oldest"
  performance:
    target_fps: 30
    max_latency_p50: 15
    max_latency_p95: 25
    max_latency_p99: 80
```

## Performance Monitoring

### Telemetry Collection
```bash
# Run 5-minute benchmark
./scripts/bench_tegrastats.sh 300

# Run 1-hour benchmark
./scripts/bench_tegrastats.sh 3600

# View results
ls -la benchmarks/
```

### Web Dashboard
```bash
# Start web interface
python3 scripts/web_interface.py

# Access dashboard
open http://localhost:8080
```

## Troubleshooting

### Common Issues

1. **Camera not detected**
   ```bash
   # Check camera devices
   ls -la /dev/video*
   
   # Test camera with GStreamer
   gst-launch-1.0 v4l2src device=/dev/video0 ! autovideosink
   ```

2. **TensorRT engine not found**
   ```bash
   # Build engines
   ./scripts/build_engines.sh
   
   # Check engine files
   ls -la models/*.trt
   ```

3. **Performance issues**
   ```bash
   # Set maximum performance
   sudo ./scripts/run_jetson_clocks.sh max
   
   # Monitor with tegrastats
   tegrastats --interval 1000
   ```

4. **Memory issues**
   ```bash
   # Check memory usage
   free -h
   
   # Clear swap
   sudo swapoff -a && sudo swapon -a
   ```

### Log Analysis
```bash
# View application logs
tail -f logs/plowpilot.log

# View system logs
sudo journalctl -u plowpilot -f

# View Docker logs
docker-compose logs -f plowpilot
```

## Performance Benchmarks

### Target Performance (P1)
- **FPS**: 720p @ 30 FPS sustained (≥28 FPS over 10 min)
- **Latency**: ≤15 ms p50 / ≤25 ms p95
- **CPU**: ≤25% of one core
- **Memory**: No leaks (ΔRSS < 50 MB over 30 min)

### Target Performance (P2)
- **FPS**: 720p @ 30 FPS end-to-end
- **Latency**: P95 ≤80 ms
- **Inference**: ≤20-25 ms (YOLOv8n FP16)
- **GPU**: 60-90% utilization, no queue growth

### Target Performance (P3)
- **Recording**: 720p @ 20-30 FPS without starving inference
- **Alerts**: ≤500 ms from event occurrence
- **Reliability**: 24h soak, zero crashes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NVIDIA Jetson](https://developer.nvidia.com/embedded/jetson-orin-nano) for the hardware platform
- [TensorRT](https://developer.nvidia.com/tensorrt) for inference optimization
- [GStreamer](https://gstreamer.freedesktop.org/) for multimedia pipeline
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for computer vision

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

---

**PlowPilot AI-Vision** - Real-time video analytics for autonomous snow plow operations
