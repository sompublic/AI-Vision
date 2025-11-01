# PlowPilot AI-Vision - Immediate Next Steps

## 🎯 Critical Path (This Week)

### 1. Fix CUDA Headers - BLOCKER ⚠️
**Priority**: CRITICAL  
**Estimated Time**: 1-2 hours  
**Status**: Blocking Phase 2 completion

**Action Items**:
```bash
# Check if CUDA headers exist
ls -la /usr/local/cuda*/targets/aarch64-linux/include/cuda_runtime_api.h

# If missing, install (JetPack should include, but verify)
sudo apt-get update
sudo apt-get install cuda-toolkit-12-6  # Adjust version as needed

# Verify TensorRT can find CUDA
cd build
cmake ..
# Check output for CUDA_INCLUDE_DIR
```

**Expected Outcome**: TensorRT can allocate GPU memory, inference uses GPU

---

### 2. Complete Main Pipeline Integration ⚠️
**Priority**: HIGH  
**Estimated Time**: 4-6 hours  
**Status**: Components exist but not connected

**Action Items**:
- [ ] Update `src/main.cpp` to instantiate AsyncPipeline
- [ ] Connect command-line arguments to phase selection
- [ ] Implement P1 mode (capture + display)
- [ ] Implement P2 mode (capture + inference + display)
- [ ] Implement P3 mode (full pipeline with recording)
- [ ] Add proper signal handling and graceful shutdown
- [ ] Test each phase independently

**Files to Modify**:
- `src/main.cpp` - Add component instantiation and connection
- `src/pipeline.cpp` - Ensure AsyncPipeline exposes necessary methods

**Test Commands**:
```bash
cd build
./plowpilot --phase P1  # Capture only
./plowpilot --phase P2  # Capture + inference
./plowpilot --phase P3  # Full pipeline
```

---

### 3. Fix TensorRT Inference Implementation ⚠️
**Priority**: HIGH  
**Estimated Time**: 6-8 hours  
**Status**: Using CPU stub, needs GPU implementation

**Action Items**:
- [ ] Replace `malloc()` with `cudaMalloc()` in `allocateMemory()`
- [ ] Implement CUDA memory transfers (`cudaMemcpy`)
- [ ] Replace dummy inference with actual TensorRT execution
- [ ] Implement proper YOLOv8 output parsing
- [ ] Add NMS (Non-Maximum Suppression) for detection filtering
- [ ] Test with real camera frames

**Files to Modify**:
- `src/infer_trt.cpp` - `allocateMemory()`, `performInference()`

**Key Changes**:
```cpp
// Replace this:
d_input_ = malloc(input_size);

// With this:
cudaMalloc(&d_input_, input_size);
cudaMemcpy(d_input_, normalized.data, input_size, cudaMemcpyHostToDevice);

// Then execute TensorRT:
context_->enqueueV2(bindings, stream, nullptr);
```

---

## 📊 Validation & Testing (Week 2)

### 4. Implement Performance Monitoring 📊
**Priority**: MEDIUM  
**Estimated Time**: 4-6 hours

**Action Items**:
- [ ] Add latency measurement (capture → display)
- [ ] Calculate percentiles (p50, p95, p99)
- [ ] Add CPU usage monitoring (per-thread)
- [ ] Add GPU usage monitoring (tegrastats integration)
- [ ] Implement memory tracking (RSS delta)
- [ ] Create performance log file

**New Files**:
- `src/telemetry.cpp` - Performance monitoring class
- `src/metrics.cpp` - Metrics collection utilities

---

### 5. Run Performance Validation Tests 🧪
**Priority**: MEDIUM  
**Estimated Time**: 2-4 hours (plus test runtime)

**Test Scenarios**:
```bash
# Test 1: 10-minute FPS stability
./plowpilot --phase P1 &
./scripts/bench_tegrastats.sh 600  # 10 minutes

# Test 2: 30-minute memory leak
./plowpilot --phase P1 &
# Monitor RSS every 5 minutes
watch -n 300 'ps aux | grep plowpilot | awk "{print \$6}"'

# Test 3: Latency measurement
./plowpilot --phase P2 --enable-telemetry
# Analyze latency logs

# Test 4: GPU utilization
./plowpilot --phase P2 &
tegrastats --interval 1000
```

---

## 🔧 Optimization (Week 2-3)

### 6. Camera Performance Investigation 📷
**Priority**: MEDIUM  
**Estimated Time**: 2-3 hours

**Action Items**:
- [ ] Test camera format capabilities:
  ```bash
  v4l2-ctl --list-formats-ext --device=/dev/video0
  ```
- [ ] Try different GStreamer pipeline configurations
- [ ] Test DMABUF zero-copy if supported
- [ ] Document actual camera limitations
- [ ] Adjust performance targets if hardware-limited

---

### 7. Audio Sync Optimization 🎵
**Priority**: LOW  
**Estimated Time**: 3-4 hours

**Action Items**:
- [ ] Research GStreamer audio sync solutions
- [ ] Test different container formats (MKV, MP4)
- [ ] Implement timestamp-based synchronization
- [ ] Consider separate audio/video files with sync metadata
- [ ] Validate sync during playback

---

## 🚀 Production Readiness (Week 3-4)

### 8. End-to-End Testing 🧪
**Priority**: MEDIUM  
**Estimated Time**: 1 day + test runtime

**Test Scenarios**:
- [ ] 24-hour soak test (zero crashes target)
- [ ] Camera disconnect recovery test
- [ ] Recording stability test (no inference starvation)
- [ ] MQTT alert latency test (if broker available)
- [ ] Memory leak validation (30+ minutes)

---

### 9. Systemd Service Deployment 🏭
**Priority**: LOW  
**Estimated Time**: 2-3 hours

**Action Items**:
```bash
# Install services
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable plowpilot
sudo systemctl enable plowpilot-clocks

# Test
sudo systemctl start plowpilot
sudo systemctl status plowpilot
sudo journalctl -u plowpilot -f
```

---

## 📋 Quick Reference Checklist

### Immediate (Do First)
- [ ] Install CUDA headers
- [ ] Verify TensorRT can use GPU
- [ ] Integrate main pipeline in `main.cpp`
- [ ] Fix TensorRT GPU memory allocation
- [ ] Test P1 mode end-to-end

### Short-term (This Week)
- [ ] Test P2 mode (capture + inference)
- [ ] Test P3 mode (full pipeline)
- [ ] Add performance monitoring
- [ ] Run 10-minute stability test

### Medium-term (Next Week)
- [ ] Run 24-hour soak test
- [ ] Fix audio sync (if critical)
- [ ] Deploy systemd services
- [ ] Document deployment process

---

## 🔍 Debugging Commands

### Check CUDA Installation
```bash
# Find CUDA headers
find /usr/local -name "cuda_runtime_api.h" 2>/dev/null
find /usr/include -name "cuda_runtime_api.h" 2>/dev/null

# Check CUDA version
nvcc --version

# Check TensorRT
dpkg -l | grep tensorrt
```

### Test Components Individually
```bash
# Test capture
cd build
./test_capture

# Test inference (if CUDA fixed)
./test_inference

# Test recording
./test_recording

# Test pipeline
./test_pipeline
```

### Monitor Performance
```bash
# System resources
htop
tegrastats --interval 1000

# Process-specific
ps aux | grep plowpilot
top -p $(pgrep plowpilot)
```

---

## 📞 Getting Help

### Common Issues
1. **CUDA headers not found**: Check JetPack installation
2. **TensorRT engine not loading**: Verify engine file path
3. **Camera not detected**: Check `/dev/video*` devices
4. **Low FPS**: May be hardware limitation, check camera format support

### Resources
- NVIDIA Jetson Developer Forums
- TensorRT Documentation
- GStreamer Documentation
- Project README.md and CONTEXT_SUMMARY.md

---

**Last Updated**: 2025-01-25  
**Next Review**: After completing critical path items
