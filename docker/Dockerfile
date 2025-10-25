# PlowPilot AI-Vision Dockerfile
# Multi-stage build for NVIDIA Jetson Orin Nano
# Builder stage: ONNX export and model preparation
# Runtime stage: TensorRT inference and GStreamer pipeline

# =============================================================================
# BUILDER STAGE - Model Export and ONNX Conversion
# =============================================================================
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-ugly1.0-dev \
    libgstreamer1.0-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-good1.0-0 \
    libgstreamer-plugins-ugly1.0-0 \
    libgstreamer1.0-tools \
    libgstreamer1.0-x \
    libgstreamer1.0-alsa \
    libgstreamer1.0-gl \
    libgstreamer1.0-gtk3 \
    libgstreamer1.0-qt5 \
    libgstreamer1.0-pulseaudio \
    libopencv-dev \
    libopencv-contrib-dev \
    python3-opencv \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for model export
RUN pip3 install --no-cache-dir \
    ultralytics \
    onnx \
    onnxruntime \
    numpy \
    opencv-python \
    pillow \
    pyyaml

# Set working directory
WORKDIR /workspace

# Copy model export script
COPY scripts/export_model.py /workspace/
COPY configs/model.yaml /workspace/

# Export YOLOv8n to ONNX
RUN python3 export_model.py

# =============================================================================
# RUNTIME STAGE - TensorRT Inference and GStreamer Pipeline
# =============================================================================
FROM nvcr.io/nvidia/l4t-tensorrt:r35.2.1-trt8.6.1 as runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TENSORRT_ROOT=/usr/src/tensorrt
ENV PATH=${TENSORRT_ROOT}/bin:${PATH}
ENV LD_LIBRARY_PATH=${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-ugly1.0-dev \
    libgstreamer1.0-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-good1.0-0 \
    libgstreamer-plugins-ugly1.0-0 \
    libgstreamer1.0-tools \
    libgstreamer1.0-x \
    libgstreamer1.0-alsa \
    libgstreamer1.0-gl \
    libgstreamer1.0-gtk3 \
    libgstreamer1.0-qt5 \
    libgstreamer1.0-pulseaudio \
    libopencv-dev \
    libopencv-contrib-dev \
    python3-opencv \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    libyaml-cpp-dev \
    libpaho-mqtt-dev \
    libpaho-mqttpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    pyyaml \
    paho-mqtt

# Create application user
RUN useradd -m -s /bin/bash plowpilot && \
    usermod -aG video plowpilot && \
    usermod -aG audio plowpilot

# Set working directory
WORKDIR /app

# Copy application files
COPY --from=builder /workspace/models/ /app/models/
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY CMakeLists.txt /app/
COPY .env /app/.env

# Create necessary directories
RUN mkdir -p /app/data /app/recordings /app/logs /app/benchmarks && \
    chown -R plowpilot:plowpilot /app

# Build the application
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# Copy systemd service
COPY systemd/plowpilot.service /etc/systemd/system/

# Set permissions
RUN chmod +x /app/scripts/*.sh && \
    chown -R plowpilot:plowpilot /app

# Switch to application user
USER plowpilot

# Expose ports (if needed for MQTT, web interface, etc.)
EXPOSE 1883 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep -f plowpilot || exit 1

# Default command
CMD ["/app/build/plowpilot", "--config", "/app/configs/pipeline.yaml"]

# =============================================================================
# DEVELOPMENT STAGE - For development and debugging
# =============================================================================
FROM runtime as development

# Install development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    strace \
    htop \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Python development dependencies
RUN pip3 install --no-cache-dir \
    ipython \
    jupyter \
    matplotlib \
    pandas \
    seaborn

# Set development environment
ENV PLOWPILOT_DEBUG=1
ENV PLOWPILOT_LOG_LEVEL=DEBUG

# Override command for development
CMD ["/bin/bash"]

# =============================================================================
# BUILD STAGE - For building TensorRT engines
# =============================================================================
FROM runtime as tensorrt-builder

# Install TensorRT tools
RUN apt-get update && apt-get install -y \
    tensorrt \
    python3-libnvinfer-dev \
    python3-libnvinfer \
    && rm -rf /var/lib/apt/lists/*

# Copy build scripts
COPY scripts/build_engines.sh /app/scripts/
COPY scripts/run_jetson_clocks.sh /app/scripts/
COPY scripts/bench_tegrastats.sh /app/scripts/

# Set permissions
RUN chmod +x /app/scripts/*.sh

# Override command for TensorRT engine building
CMD ["/app/scripts/build_engines.sh", "--int8", "--benchmark"]
