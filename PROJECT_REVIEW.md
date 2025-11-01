# PlowPilot AI-Vision - Project Review & Next Steps
**Date**: 2025-01-25  
**Project**: Real-time video analytics pipeline on NVIDIA Jetson Orin Nano  
**Status**: Phase 1 Complete, Phase 2 Partial, Phase 3 Partial

---

## 📊 Executive Summary

**Overall Progress**: ~60% Complete
- ✅ **Phase 1 (Capture)**: ~90% - Core functionality working, performance validation pending
- ⚠️ **Phase 2 (Inference)**: ~40% - TensorRT engine built, integration incomplete
- ✅ **Phase 3 (Recording/Events)**: ~85% - Recording working, MQTT pending

**Critical Blockers**:
1. CUDA headers missing - preventing full TensorRT integration
2. Main pipeline integration incomplete - components exist but not unified
3. Performance validation pending - metrics collection not implemented

---

## ✅ Completed Components

### 1. Build System & Infrastructure
- ✅ CMake build system configured for Jetson Orin Nano
- ✅ Dependency management (OpenCV, GStreamer, TensorRT)
- ✅ Stub implementations for yaml-cpp and MQTT
- ✅ Docker containerization support
- ✅ Systemd service files

### 2. Phase 1: Capture System (P1_capture_profile)
- ✅ **GStreamerCapture** (`capture_gst.cpp`)
  - Camera capture working at 14.4 FPS (640x480 YUYV)
  - GStreamer pipeline configured
  - Frame buffering and drop-oldest policy
  - ⚠️ Camera format limitation: YUYV only (MJPG not supported)
  - ⚠️ Performance: 14.4 FPS vs target 28+ FPS @ 720p30

### 3. Phase 2: Inference System (P2_infer_async)
- ✅ **TensorRT Engine**: Built successfully (8.9 MB FP16)
  - Model: YOLOv8n → ONNX → TensorRT FP16
  - Build script with timeout protection
  - ⚠️ **TensorRTInference** (`infer_trt.cpp`): Stub implementation
    - Missing CUDA headers preventing GPU memory allocation
    - Using CPU memory as placeholder
    - Inference loop architecture complete
    - Actual TensorRT execution not working

### 4. Phase 3: Recording & Events (P3_record_events_edge)
- ✅ **GStreamerRecorder** (`record_gst.cpp`)
  - Video recording: x264 encoding working
  - Audio recording: MP3 encoding implemented (48kHz mono)
  - File chunking: Duration + size-based working
  - Container: AVI format with H.264 video + MP3 audio
  - ⚠️ Audio sync issues identified
- ✅ **EventDetector** (`events_mqtt.cpp`)
  - Event detection framework implemented
  - MQTT client stub (no broker available)
  - Alert system architecture ready

### 5. Pipeline Architecture
- ✅ **AsyncPipeline** (`pipeline.cpp`)
  - Multi-threaded architecture designed
  - Bounded queues with drop-oldest policy
  - Producer/consumer pattern
  - ⚠️ Not integrated into main application

### 6. Configuration System
- ✅ YAML configuration files (camera, model, pipeline)
- ✅ Stub YAML parser for environments without yaml-cpp
- ✅ Comprehensive settings (audio, video, performance)

### 7. Scripts & Utilities
- ✅ `build_engines.sh` - TensorRT engine builder with timeout protection
- ✅ `run_jetson_clocks.sh` - Performance tuning
- ✅ `bench_tegrastats.sh` - Telemetry collection
- ✅ `export_model.py` - YOLOv8n to ONNX export
- ✅ `web_interface.py` - Web dashboard (not tested)

---

## ⚠️ Partial/Incomplete Components

### 1. Main Application Integration
- ⚠️ **main.cpp**: Skeleton implementation
  - Phase selection (P1/P2/P3) not implemented
  - Main loop placeholder
  - Components not instantiated or connected

### 2. TensorRT Inference
- ⚠️ **CUDA Headers Missing**: 
  - CMake finds TensorRT but not CUDA headers
  - GPU memory allocation using `malloc()` (CPU memory)
  - Actual TensorRT execution blocked
- ⚠️ **Inference Implementation**:
  - Engine loading works
  - Preprocessing implemented
  - Post-processing (NMS, detection parsing) incomplete
  - Using dummy detections for testing

### 3. Performance Monitoring
- ⚠️ **Telemetry**: Framework exists but not validated
  - FPS tracking implemented but not tested
  - Latency measurement (p50/p95/p99) not implemented
  - CPU/GPU usage monitoring not integrated
  - Memory leak detection not automated

### 4. Audio Sync
- ⚠️ **Audio-Video Synchronization**: 
  - Both streams recorded correctly
  - Timing synchronization issues during playback
  - May need separate files with sync metadata

### 5. MQTT Integration
- ⚠️ **MQTT Broker**: Not available
  - Stub implementation working
  - Real MQTT alerts cannot be tested
  - Connection/reconnection logic not validated

---

## 🚫 Missing/Blocked Components

### 1. Critical Dependencies
- ❌ **CUDA Development Headers**: Required for TensorRT GPU memory
  - Path: `/usr/local/cuda-12.6/targets/aarch64-linux/include`
  - Impact: Inference cannot use GPU, uses CPU fallback

### 2. Integration Work
- ❌ **Main Pipeline Integration**: Components exist but not connected
- ❌ **Performance Validation**: Metrics collection not tested
- ❌ **End-to-End Testing**: Full pipeline not validated

### 3. Production Readiness
- ❌ **Error Recovery**: Camera disconnect handling not tested
- ❌ **24h Soak Test**: Long-term stability not validated
- ❌ **Monitoring Dashboard**: Web interface not tested

---

## 📈 Current Performance Metrics

### Achieved vs Targets

| Component | Target | Achieved | Status |
|-----------|--------|----------|--------|
| **Camera FPS** | ≥28 FPS @ 720p30 | 14.4 FPS @ 480p15 | ⚠️ Below target |
| **Capture Latency** | p50≤15ms, p95≤25ms | Not measured | ❌ Pending |
| **CPU Usage** | ≤25% single core | Not measured | ❌ Pending |
| **Memory Stability** | ΔRSS <50MB/30min | Not measured | ❌ Pending |
| **Inference Time** | ≤20-25ms | Not measured | ❌ Blocked (CUDA) |
| **End-to-End Latency** | p95≤80ms | Not measured | ❌ Blocked |
| **Recording FPS** | 20-30 FPS | Working | ✅ |
| **Alert Latency** | ≤500ms | Not tested | ⚠️ Pending MQTT |

### Known Issues
1. **Camera Format**: Limited to YUYV (640x480), cannot achieve 720p30
2. **FPS**: Currently 14.4 FPS vs 28+ FPS target
3. **Audio Sync**: Timing issues between audio/video streams
4. **CUDA**: Missing headers prevent GPU inference

---

## 🎯 Next Steps Roadmap

### Immediate Priority (Week 1)

#### 1. Fix CUDA Headers Issue ⚠️ CRITICAL
**Goal**: Enable full TensorRT GPU inference
- [ ] Check CUDA installation on Jetson
- [ ] Install CUDA development headers if missing
- [ ] Verify TensorRT can allocate GPU memory
- [ ] Test actual GPU inference with real frames

**Commands**:
```bash
# Check CUDA installation
ls -la /usr/local/cuda*/targets/aarch64-linux/include/cuda_runtime_api.h

# Install if missing (JetPack should include this)
sudo apt-get update
sudo apt-get install cuda-toolkit-12-6
```

#### 2. Complete Main Pipeline Integration ⚠️ HIGH
**Goal**: Connect all components in main.cpp
- [ ] Implement P1 mode (capture + display only)
- [ ] Implement P2 mode (capture + inference + display)
- [ ] Implement P3 mode (full pipeline with recording)
- [ ] Add proper error handling and shutdown
- [ ] Connect AsyncPipeline to main application

**Files to modify**:
- `src/main.cpp` - Main application logic
- `src/pipeline.cpp` - Ensure proper component integration

#### 3. Fix TensorRT Inference Implementation ⚠️ HIGH
**Goal**: Replace stub with real TensorRT execution
- [ ] Replace CPU `malloc()` with CUDA `cudaMalloc()`
- [ ] Implement proper GPU memory transfers
- [ ] Add CUDA kernel for preprocessing (if needed)
- [ ] Implement TensorRT execution context
- [ ] Parse YOLOv8 output format correctly
- [ ] Implement NMS (Non-Maximum Suppression)
- [ ] Add detection result parsing

**Files to modify**:
- `src/infer_trt.cpp` - Replace stub implementation

### Short-term (Week 2)

#### 4. Performance Validation & Telemetry 📊
**Goal**: Measure and validate performance targets
- [ ] Implement latency measurement (p50/p95/p99)
- [ ] Add CPU/GPU usage monitoring
- [ ] Implement memory leak detection
- [ ] Create performance logging system
- [ ] Run 10-minute stability tests
- [ ] Run 30-minute memory leak tests

**New Files**:
- `src/telemetry.cpp` - Performance monitoring
- `src/metrics.cpp` - Metrics collection

#### 5. Camera Performance Optimization 🔧
**Goal**: Achieve target 28+ FPS @ 720p30
- [ ] Investigate camera format options
- [ ] Test different GStreamer pipeline configurations
- [ ] Optimize buffer sizes and drop policies
- [ ] Consider hardware acceleration options
- [ ] Test DMABUF zero-copy if available

**If camera hardware limitation**:
- [ ] Document actual capabilities
- [ ] Adjust targets to match hardware (e.g., 480p30)

#### 6. Audio Sync Fix 🎵
**Goal**: Fix audio-video synchronization
- [ ] Research GStreamer audio sync solutions
- [ ] Test different container formats (MP4, MKV)
- [ ] Consider separate audio/video files with sync metadata
- [ ] Implement timestamp-based synchronization
- [ ] Validate sync during playback

### Medium-term (Week 3-4)

#### 7. Complete Phase 2 Integration 🚀
**Goal**: Full inference pipeline working
- [ ] End-to-end test: Capture → Infer → Display
- [ ] Validate p95 latency ≤80ms
- [ ] Monitor GPU utilization (60-90% target)
- [ ] Test queue stability over 10+ minutes
- [ ] Benchmark FP16 performance

#### 8. Phase 3 Completion 📹
**Goal**: Production-ready recording and events
- [ ] Test recording without starving inference
- [ ] Implement event-based recording triggers
- [ ] Test MQTT alerts (if broker available)
- [ ] Validate alert latency ≤500ms
- [ ] Test camera disconnect recovery
- [ ] Run 24-hour soak test

#### 9. Production Deployment 🏭
**Goal**: Systemd service deployment
- [ ] Test systemd services
- [ ] Configure auto-start on boot
- [ ] Set up log rotation
- [ ] Create monitoring dashboard
- [ ] Document deployment procedures

---

## 🔧 Technical Debt & Improvements

### Code Quality
- [ ] Add comprehensive error handling
- [ ] Implement proper logging system (replace cout/cerr)
- [ ] Add unit tests for core components
- [ ] Improve code documentation
- [ ] Add performance profiling tools

### Architecture
- [ ] Implement dynamic model loading
- [ ] Add support for multiple camera inputs
- [ ] Enhance MQTT reliability and reconnection
- [ ] Add web-based configuration interface
- [ ] Implement automatic model updates

### Performance
- [ ] Optimize GStreamer pipelines
- [ ] Implement dynamic batching (if needed)
- [ ] Add GPU memory pooling
- [ ] Optimize frame buffer management

---

## 📝 Configuration Status

### Working Configurations ✅
- Camera: YUYV format, 640x480 @ 15 FPS
- Recording: x264 ultrafast, AVI container
- Audio: MP3, 48kHz mono, 128kbps
- TensorRT: FP16 engine built (8.9 MB)

### Target Configurations 🎯
- Camera: 720p @ 30 FPS (hardware limitation)
- Recording: 720p @ 20-30 FPS without starving inference
- Inference: ≤20-25ms, p95 latency ≤80ms

---

## 🐛 Known Bugs & Issues

1. **Camera Format Limitation**
   - **Issue**: Camera only supports YUYV at 640x480, not MJPG at 720p
   - **Impact**: Cannot achieve target 720p30 FPS
   - **Status**: Hardware limitation, needs investigation

2. **CUDA Headers Missing**
   - **Issue**: TensorRT cannot allocate GPU memory
   - **Impact**: Inference uses CPU fallback (slow)
   - **Status**: Blocking Phase 2 completion

3. **Audio Sync Issues**
   - **Issue**: Timing synchronization between audio/video
   - **Impact**: Playback sync problems
   - **Status**: Functional but needs optimization

4. **Main Pipeline Not Integrated**
   - **Issue**: Components exist but not connected in main.cpp
   - **Impact**: Cannot run full pipeline
   - **Status**: High priority to fix

5. **Performance Metrics Not Collected**
   - **Issue**: No automated performance validation
   - **Impact**: Cannot verify acceptance criteria
   - **Status**: Needs implementation

---

## 💡 Recommendations

### Immediate Actions
1. **Install CUDA headers** - Critical blocker for inference
2. **Integrate main pipeline** - Connect existing components
3. **Fix TensorRT implementation** - Enable GPU inference
4. **Add performance monitoring** - Validate targets

### Short-term Actions
5. **Investigate camera capabilities** - May need to adjust targets
6. **Fix audio sync** - Improve recording quality
7. **Run stability tests** - Validate 24h operation

### Long-term Actions
8. **Production deployment** - Systemd services
9. **Monitoring dashboard** - Web interface
10. **Documentation** - User guides and deployment docs

---

## 📊 Progress Summary

| Phase | Component | Status | Completion |
|-------|-----------|--------|------------|
| **P1** | Capture | ✅ Working | 90% |
| **P1** | Performance Validation | ❌ Not started | 0% |
| **P2** | TensorRT Engine | ✅ Built | 100% |
| **P2** | Inference Implementation | ⚠️ Stub | 40% |
| **P2** | Pipeline Integration | ⚠️ Partial | 50% |
| **P3** | Recording | ✅ Working | 95% |
| **P3** | Audio | ⚠️ Sync issues | 80% |
| **P3** | Events | ✅ Framework | 85% |
| **P3** | MQTT | ⚠️ Stub only | 60% |
| **Integration** | Main Pipeline | ❌ Not connected | 20% |
| **Testing** | Performance | ❌ Not started | 0% |
| **Deployment** | Systemd | ⚠️ Configured | 70% |

**Overall Project Completion**: ~60%

---

## 🎯 Success Criteria Status

### Phase 1 (P1_capture_profile)
- ✅ Low-latency GStreamer capture - **Working**
- ⚠️ ≥28 FPS @ 720p30 - **14.4 FPS @ 480p15 (hardware limitation)**
- ❌ p50≤15ms, p95≤25ms latency - **Not measured**
- ❌ ≤25% single core CPU - **Not measured**
- ❌ ΔRSS <50MB/30min - **Not measured**

### Phase 2 (P2_infer_async)
- ❌ p95 ≤80ms end-to-end - **Blocked (CUDA)**
- ❌ ≤20-25ms inference - **Blocked (CUDA)**
- ❌ GPU util 60-90% - **Blocked (CUDA)**
- ❌ No queue growth 10min - **Not tested**

### Phase 3 (P3_record_events_edge)
- ✅ Recording 20-30 FPS - **Working**
- ❌ Alert ≤500ms - **Not tested (MQTT stub)**
- ❌ 24h soak zero crashes - **Not tested**
- ❌ Camera disconnect recovery - **Not tested**

---

**Next Review**: After CUDA headers installation and main pipeline integration  
**Priority**: Fix CUDA headers → Integrate pipeline → Validate performance
