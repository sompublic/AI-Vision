#!/bin/bash
# build_engines.sh - Build TensorRT engines from ONNX models
# PlowPilot AI-Vision - NVIDIA Jetson Orin Nano

set -e

# Configuration
MODEL_NAME="yolov8n"
MODELS_DIR="models"
ONNX_MODEL="${MODELS_DIR}/${MODEL_NAME}.onnx"
TRT_ENGINE_FP16="${MODELS_DIR}/${MODEL_NAME}_fp16.trt"
TRT_ENGINE_INT8="${MODELS_DIR}/${MODEL_NAME}_int8.trt"
CALIBRATION_DATA="data/calibration"
CALIBRATION_CACHE="${MODELS_DIR}/${MODEL_NAME}_int8_calibration.cache"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PlowPilot AI-Vision - TensorRT Engine Builder${NC}"
echo "=================================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${YELLOW}Warning: Not running on NVIDIA Jetson. TensorRT may not be available.${NC}"
fi

# Create models directory
mkdir -p "${MODELS_DIR}"

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: $1 not found${NC}"
        return 1
    fi
    return 0
}

# Function to download YOLOv8n ONNX model
download_yolov8n() {
    echo -e "${YELLOW}Downloading YOLOv8n ONNX model...${NC}"
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python3 not found. Please install Python3 to download the model.${NC}"
        exit 1
    fi
    
    # Create temporary Python script to download model
    cat > download_model.py << 'EOF'
import torch
from ultralytics import YOLO
import os

# Download YOLOv8n
model = YOLO('yolov8n.pt')

# Export to ONNX
model.export(format='onnx', imgsz=640, optimize=True)

print("YOLOv8n ONNX model exported successfully")
EOF
    
    python3 download_model.py
    mv yolov8n.onnx "${ONNX_MODEL}"
    rm download_model.py
    
    echo -e "${GREEN}YOLOv8n ONNX model downloaded: ${ONNX_MODEL}${NC}"
}

# Function to build FP16 engine with timeout protection
build_fp16_engine() {
    echo -e "${YELLOW}Building FP16 TensorRT engine...${NC}"
    
    # Check if trtexec is available
    if ! command -v trtexec &> /dev/null; then
        echo -e "${RED}Error: trtexec not found. Please install TensorRT.${NC}"
        exit 1
    fi
    
    # ✅ Add timeout protection (30 minutes max)
    timeout 1800 trtexec --onnx="${ONNX_MODEL}" \
            --saveEngine="${TRT_ENGINE_FP16}" \
            --fp16 \
            --workspace=2048 \
            --verbose \
            --noDataTransfers \
            --useCudaGraph \
            --minShapes=input:1x3x640x640 \
            --optShapes=input:1x3x640x640 \
            --maxShapes=input:1x3x640x640
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}FP16 engine built successfully: ${TRT_ENGINE_FP16}${NC}"
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}❌ FP16 engine build TIMED OUT after 30 minutes${NC}"
        echo -e "${YELLOW}💡 Try reducing workspace size or using a simpler model${NC}"
        exit 1
    else
        echo -e "${RED}❌ Failed to build FP16 engine (exit code: $exit_code)${NC}"
        exit 1
    fi
}

# Function to build INT8 engine
build_int8_engine() {
    echo -e "${YELLOW}Building INT8 TensorRT engine...${NC}"
    
    # Check if calibration data exists
    if [ ! -d "${CALIBRATION_DATA}" ]; then
        echo -e "${YELLOW}Calibration data not found. Creating sample calibration data...${NC}"
        mkdir -p "${CALIBRATION_DATA}"
        
        # Create a simple calibration script
        cat > create_calibration_data.py << 'EOF'
import cv2
import numpy as np
import os

# Create calibration directory
os.makedirs('data/calibration', exist_ok=True)

# Generate 100 sample images for calibration
for i in range(100):
    # Create random image
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Add some structure
    cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(img, (400, 400), 50, (128, 128, 128), -1)
    
    # Save image
    cv2.imwrite(f'data/calibration/calib_{i:03d}.jpg', img)

print("Calibration data created: 100 sample images")
EOF
        
        python3 create_calibration_data.py
        rm create_calibration_data.py
    fi
    
    # ✅ Build INT8 engine with timeout protection (45 minutes max)
    timeout 2700 trtexec --onnx="${ONNX_MODEL}" \
            --saveEngine="${TRT_ENGINE_INT8}" \
            --int8 \
            --calib="${CALIBRATION_DATA}" \
            --workspace=2048 \
            --verbose \
            --noDataTransfers \
            --useCudaGraph \
            --minShapes=input:1x3x640x640 \
            --optShapes=input:1x3x640x640 \
            --maxShapes=input:1x3x640x640
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}INT8 engine built successfully: ${TRT_ENGINE_INT8}${NC}"
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}❌ INT8 engine build TIMED OUT after 45 minutes${NC}"
        echo -e "${YELLOW}💡 INT8 calibration can take longer. Try FP16 first.${NC}"
        exit 1
    else
        echo -e "${RED}❌ Failed to build INT8 engine (exit code: $exit_code)${NC}"
        exit 1
    fi
}

# Function to benchmark engines
benchmark_engines() {
    echo -e "${YELLOW}Benchmarking engines...${NC}"
    
    if [ -f "${TRT_ENGINE_FP16}" ]; then
        echo -e "${GREEN}Benchmarking FP16 engine:${NC}"
        trtexec --loadEngine="${TRT_ENGINE_FP16}" \
                --warmUp=100 \
                --iterations=1000 \
                --verbose
    fi
    
    if [ -f "${TRT_ENGINE_INT8}" ]; then
        echo -e "${GREEN}Benchmarking INT8 engine:${NC}"
        trtexec --loadEngine="${TRT_ENGINE_INT8}" \
                --warmUp=100 \
                --iterations=1000 \
                --verbose
    fi
}

# Main execution
main() {
    echo "Starting TensorRT engine build process..."
    
    # Check if ONNX model exists, download if not
    if ! check_file "${ONNX_MODEL}"; then
        download_yolov8n
    fi
    
    # Build FP16 engine
    if ! check_file "${TRT_ENGINE_FP16}"; then
        build_fp16_engine
    else
        echo -e "${GREEN}FP16 engine already exists: ${TRT_ENGINE_FP16}${NC}"
    fi
    
    # Build INT8 engine (optional)
    if [ "$1" = "--int8" ]; then
        if ! check_file "${TRT_ENGINE_INT8}"; then
            build_int8_engine
        else
            echo -e "${GREEN}INT8 engine already exists: ${TRT_ENGINE_INT8}${NC}"
        fi
    fi
    
    # Benchmark engines
    if [ "$2" = "--benchmark" ]; then
        benchmark_engines
    fi
    
    echo -e "${GREEN}TensorRT engine build completed successfully!${NC}"
    echo ""
    echo "Available engines:"
    ls -la "${MODELS_DIR}"/*.trt 2>/dev/null || echo "No engines found"
    echo ""
    echo "Usage:"
    echo "  $0                    # Build FP16 engine only"
    echo "  $0 --int8            # Build both FP16 and INT8 engines"
    echo "  $0 --int8 --benchmark # Build engines and run benchmarks"
}

# Run main function
main "$@"
