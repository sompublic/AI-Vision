#!/bin/bash
# debug_tensorrt.sh - Comprehensive TensorRT debugging script
# PlowPilot AI-Vision - NVIDIA Jetson Orin Nano

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 PlowPilot AI-Vision - TensorRT Debugging Tool${NC}"
echo "=================================================="

# Function to check system requirements
check_system() {
    echo -e "${YELLOW}📋 Checking system requirements...${NC}"
    
    # Check if running on Jetson
    if [ -f /etc/nv_tegra_release ]; then
        echo -e "${GREEN}✓ Running on NVIDIA Jetson${NC}"
        cat /etc/nv_tegra_release
    else
        echo -e "${YELLOW}⚠️  Not running on NVIDIA Jetson${NC}"
    fi
    
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        echo -e "${GREEN}✓ CUDA available: $(nvcc --version | grep release | cut -d' ' -f5)${NC}"
    else
        echo -e "${RED}❌ CUDA not found${NC}"
    fi
    
    # Check TensorRT version
    if command -v trtexec &> /dev/null; then
        echo -e "${GREEN}✓ TensorRT available${NC}"
        trtexec --version 2>/dev/null || echo "Version info not available"
    else
        echo -e "${RED}❌ TensorRT not found${NC}"
    fi
    
    # Check GPU memory
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ GPU Memory:${NC}"
        nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
    else
        echo -e "${YELLOW}⚠️  nvidia-smi not available${NC}"
    fi
}

# Function to validate ONNX model
validate_onnx() {
    echo -e "${YELLOW}🔍 Validating ONNX model...${NC}"
    
    local onnx_model="models/yolov8n.onnx"
    
    if [ ! -f "$onnx_model" ]; then
        echo -e "${RED}❌ ONNX model not found: $onnx_model${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ ONNX model found: $onnx_model${NC}"
    
    # Check file size
    local size=$(du -h "$onnx_model" | cut -f1)
    echo -e "${GREEN}✓ Model size: $size${NC}"
    
    # Validate with Python if available
    if command -v python3 &> /dev/null; then
        echo -e "${YELLOW}🔍 Validating ONNX model with Python...${NC}"
        python3 -c "
import onnx
try:
    model = onnx.load('$onnx_model')
    onnx.checker.check_model(model)
    print('✓ ONNX model is valid')
    
    # Print input/output info
    for input in model.graph.input:
        print(f'Input: {input.name}, Shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}')
    for output in model.graph.output:
        print(f'Output: {output.name}, Shape: {[d.dim_value for d in output.type.tensor_type.shape.dim]}')
        
except Exception as e:
    print(f'❌ ONNX validation failed: {e}')
    exit(1)
"
    fi
}

# Function to test TensorRT engine building with minimal settings
test_minimal_build() {
    echo -e "${YELLOW}🧪 Testing minimal TensorRT engine build...${NC}"
    
    local onnx_model="models/yolov8n.onnx"
    local test_engine="models/test_minimal.trt"
    
    if [ ! -f "$onnx_model" ]; then
        echo -e "${RED}❌ ONNX model not found for testing${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Building minimal engine (FP32, small workspace)...${NC}"
    
    # Test with minimal settings first
    timeout 600 trtexec --onnx="$onnx_model" \
            --saveEngine="$test_engine" \
            --workspace=512 \
            --verbose \
            --noDataTransfers \
            --minShapes=input:1x3x640x640 \
            --optShapes=input:1x3x640x640 \
            --maxShapes=input:1x3x640x640
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Minimal engine build successful${NC}"
        rm -f "$test_engine"  # Clean up test file
        return 0
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}❌ Minimal engine build TIMED OUT${NC}"
        echo -e "${YELLOW}💡 This indicates a fundamental issue with the ONNX model${NC}"
        return 1
    else
        echo -e "${RED}❌ Minimal engine build failed (exit code: $exit_code)${NC}"
        return 1
    fi
}

# Function to check for common issues
check_common_issues() {
    echo -e "${YELLOW}🔍 Checking for common issues...${NC}"
    
    # Check disk space
    local available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 1048576 ]; then  # Less than 1GB
        echo -e "${RED}❌ Low disk space: ${available_space}KB available${NC}"
    else
        echo -e "${GREEN}✓ Sufficient disk space: ${available_space}KB available${NC}"
    fi
    
    # Check memory
    local available_memory=$(free -m | awk 'NR==2{print $7}')
    if [ "$available_memory" -lt 1024 ]; then  # Less than 1GB
        echo -e "${RED}❌ Low available memory: ${available_memory}MB${NC}"
    else
        echo -e "${GREEN}✓ Sufficient memory: ${available_memory}MB available${NC}"
    fi
    
    # Check for running processes that might interfere
    local tensorrt_processes=$(pgrep -f trtexec || true)
    if [ -n "$tensorrt_processes" ]; then
        echo -e "${YELLOW}⚠️  Found running TensorRT processes: $tensorrt_processes${NC}"
    else
        echo -e "${GREEN}✓ No conflicting TensorRT processes${NC}"
    fi
}

# Function to provide recommendations
provide_recommendations() {
    echo -e "${BLUE}💡 Recommendations:${NC}"
    echo ""
    echo "1. **Start with FP16 only**: Use './scripts/build_engines.sh' (no --int8 flag)"
    echo "2. **Monitor system resources**: Watch GPU memory and CPU usage during build"
    echo "3. **Check logs**: Use 'journalctl -u plowpilot-tensorrt.service -f' to monitor"
    echo "4. **Test incrementally**: Build FP16 first, then INT8 if needed"
    echo "5. **Use timeout protection**: The updated scripts now include timeout protection"
    echo ""
    echo "**If builds still hang:**"
    echo "- Reduce workspace size in build_engines.sh"
    echo "- Try building on a different Jetson device"
    echo "- Check for hardware issues (thermal throttling, power supply)"
    echo "- Consider using a pre-built engine from NVIDIA"
}

# Main execution
main() {
    check_system
    echo ""
    
    check_common_issues
    echo ""
    
    if validate_onnx; then
        echo ""
        if test_minimal_build; then
            echo -e "${GREEN}🎉 System appears ready for TensorRT engine building!${NC}"
        else
            echo -e "${RED}❌ Found issues with TensorRT engine building${NC}"
        fi
    else
        echo -e "${RED}❌ ONNX model validation failed${NC}"
    fi
    
    echo ""
    provide_recommendations
}

# Run main function
main "$@"

