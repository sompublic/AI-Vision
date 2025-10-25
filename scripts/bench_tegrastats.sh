#!/bin/bash
# bench_tegrastats.sh - Benchmark and collect telemetry data
# PlowPilot AI-Vision - NVIDIA Jetson Orin Nano

set -e

# Configuration
BENCHMARK_DURATION=${1:-300}  # Default 5 minutes
OUTPUT_DIR="benchmarks"
LOG_FILE="${OUTPUT_DIR}/tegrastats_$(date +%Y%m%d_%H%M%S).log"
CSV_FILE="${OUTPUT_DIR}/telemetry_$(date +%Y%m%d_%H%M%S).csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}PlowPilot AI-Vision - Telemetry Benchmark${NC}"
echo "=============================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${RED}Error: Not running on NVIDIA Jetson${NC}"
    exit 1
fi

# Check if tegrastats is available
if ! command -v tegrastats &> /dev/null; then
    echo -e "${RED}Error: tegrastats not found. Please install JetPack.${NC}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Function to collect system info
collect_system_info() {
    echo -e "${YELLOW}Collecting system information...${NC}"
    
    cat > "${OUTPUT_DIR}/system_info.txt" << EOF
PlowPilot AI-Vision System Information
=====================================
Date: $(date)
Hostname: $(hostname)
Kernel: $(uname -r)
JetPack Version: $(cat /etc/nv_tegra_release 2>/dev/null || echo "Unknown")

CPU Information:
$(lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket")

Memory Information:
$(free -h)

GPU Information:
$(nvidia-smi 2>/dev/null || echo "nvidia-smi not available")

Storage Information:
$(df -h)

Network Information:
$(ip addr show | grep -E "inet |UP")
EOF
    
    echo -e "${GREEN}System information saved to: ${OUTPUT_DIR}/system_info.txt${NC}"
}

# Function to run benchmark
run_benchmark() {
    echo -e "${YELLOW}Starting ${BENCHMARK_DURATION}s benchmark...${NC}"
    echo "Output: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    echo ""
    
    # Start tegrastats in background
    tegrastats --interval 1000 --logfile "${LOG_FILE}" &
    TEGRASTATS_PID=$!
    
    # Wait for benchmark duration
    echo -e "${BLUE}Benchmark running... (${BENCHMARK_DURATION}s)${NC}"
    sleep "${BENCHMARK_DURATION}"
    
    # Stop tegrastats
    kill $TEGRASTATS_PID 2>/dev/null || true
    wait $TEGRASTATS_PID 2>/dev/null || true
    
    echo -e "${GREEN}Benchmark completed${NC}"
}

# Function to parse tegrastats log and create CSV
parse_tegrastats() {
    echo -e "${YELLOW}Parsing tegrastats data...${NC}"
    
    if [ ! -f "${LOG_FILE}" ]; then
        echo -e "${RED}Error: tegrastats log file not found: ${LOG_FILE}${NC}"
        return 1
    fi
    
    # Create CSV header
    cat > "${CSV_FILE}" << EOF
timestamp,cpu_freq_mhz,gpu_freq_mhz,emc_freq_mhz,ram_usage_mb,swap_usage_mb,ram_usage_percent,swap_usage_percent,gpu_util_percent,emc_util_percent,thermal_zone0_temp,thermal_zone1_temp,thermal_zone2_temp,thermal_zone3_temp,thermal_zone4_temp,thermal_zone5_temp,thermal_zone6_temp,thermal_zone7_temp
EOF
    
    # Parse tegrastats log
    awk '
    BEGIN {
        OFS = ","
    }
    {
        # Extract timestamp
        timestamp = $1 " " $2
        
        # Extract CPU frequency (MHz)
        cpu_freq = 0
        if (match($0, /CPU@([0-9]+)/)) {
            cpu_freq = substr($0, RSTART+4, RLENGTH-5)
        }
        
        # Extract GPU frequency (MHz)
        gpu_freq = 0
        if (match($0, /GR3D_FREQ ([0-9]+)/)) {
            gpu_freq = substr($0, RSTART+11, RLENGTH-12)
        }
        
        # Extract EMC frequency (MHz)
        emc_freq = 0
        if (match($0, /EMC_FREQ ([0-9]+)/)) {
            emc_freq = substr($0, RSTART+10, RLENGTH-11)
        }
        
        # Extract RAM usage
        ram_usage = 0
        ram_percent = 0
        if (match($0, /RAM ([0-9]+)\/([0-9]+)MB/)) {
            ram_usage = substr($0, RSTART+4, RLENGTH-5)
            ram_total = substr($0, RSTART+6, RLENGTH-7)
            ram_percent = (ram_usage / ram_total) * 100
        }
        
        # Extract swap usage
        swap_usage = 0
        swap_percent = 0
        if (match($0, /SWAP ([0-9]+)\/([0-9]+)MB/)) {
            swap_usage = substr($0, RSTART+5, RLENGTH-6)
            swap_total = substr($0, RSTART+7, RLENGTH-8)
            swap_percent = (swap_usage / swap_total) * 100
        }
        
        # Extract GPU utilization
        gpu_util = 0
        if (match($0, /GR3D_FREQ [0-9]+% ([0-9]+)/)) {
            gpu_util = substr($0, RSTART+11, RLENGTH-12)
        }
        
        # Extract EMC utilization
        emc_util = 0
        if (match($0, /EMC_FREQ [0-9]+% ([0-9]+)/)) {
            emc_util = substr($0, RSTART+10, RLENGTH-11)
        }
        
        # Extract thermal zone temperatures
        temp0 = temp1 = temp2 = temp3 = temp4 = temp5 = temp6 = temp7 = 0
        if (match($0, /thermal_zone0 ([0-9]+)/)) {
            temp0 = substr($0, RSTART+13, RLENGTH-14)
        }
        if (match($0, /thermal_zone1 ([0-9]+)/)) {
            temp1 = substr($0, RSTART+13, RLENGTH-14)
        }
        if (match($0, /thermal_zone2 ([0-9]+)/)) {
            temp2 = substr($0, RSTART+13, RLENGTH-14)
        }
        if (match($0, /thermal_zone3 ([0-9]+)/)) {
            temp3 = substr($0, RSTART+13, RLENGTH-14)
        }
        if (match($0, /thermal_zone4 ([0-9]+)/)) {
            temp4 = substr($0, RSTART+13, RLENGTH-14)
        }
        if (match($0, /thermal_zone5 ([0-9]+)/)) {
            temp5 = substr($0, RSTART+13, RLENGTH-14)
        }
        if (match($0, /thermal_zone6 ([0-9]+)/)) {
            temp6 = substr($0, RSTART+13, RLENGTH-14)
        }
        if (match($0, /thermal_zone7 ([0-9]+)/)) {
            temp7 = substr($0, RSTART+13, RLENGTH-14)
        }
        
        # Output CSV line
        print timestamp, cpu_freq, gpu_freq, emc_freq, ram_usage, swap_usage, ram_percent, swap_percent, gpu_util, emc_util, temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7
    }
    ' "${LOG_FILE}" >> "${CSV_FILE}"
    
    echo -e "${GREEN}CSV data saved to: ${CSV_FILE}${NC}"
}

# Function to generate summary report
generate_summary() {
    echo -e "${YELLOW}Generating summary report...${NC}"
    
    if [ ! -f "${CSV_FILE}" ]; then
        echo -e "${RED}Error: CSV file not found: ${CSV_FILE}${NC}"
        return 1
    fi
    
    # Create summary report
    cat > "${OUTPUT_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt" << EOF
PlowPilot AI-Vision Benchmark Summary
====================================
Date: $(date)
Duration: ${BENCHMARK_DURATION} seconds
Log File: ${LOG_FILE}
CSV File: ${CSV_FILE}

System Performance:
- CPU Frequency: $(awk -F',' 'NR>1 {sum+=$2; count++} END {print sum/count " MHz average"}' "${CSV_FILE}")
- GPU Frequency: $(awk -F',' 'NR>1 {sum+=$3; count++} END {print sum/count " MHz average"}' "${CSV_FILE}")
- EMC Frequency: $(awk -F',' 'NR>1 {sum+=$4; count++} END {print sum/count " MHz average"}' "${CSV_FILE}")

Memory Usage:
- RAM Usage: $(awk -F',' 'NR>1 {sum+=$5; count++} END {print sum/count " MB average"}' "${CSV_FILE}")
- RAM Percentage: $(awk -F',' 'NR>1 {sum+=$7; count++} END {print sum/count "% average"}' "${CSV_FILE}")
- Swap Usage: $(awk -F',' 'NR>1 {sum+=$6; count++} END {print sum/count " MB average"}' "${CSV_FILE}")

GPU Utilization:
- GPU Utilization: $(awk -F',' 'NR>1 {sum+=$9; count++} END {print sum/count "% average"}' "${CSV_FILE}")
- EMC Utilization: $(awk -F',' 'NR>1 {sum+=$10; count++} END {print sum/count "% average"}' "${CSV_FILE}")

Temperature:
- Thermal Zone 0: $(awk -F',' 'NR>1 {sum+=$11; count++} END {print sum/count "°C average"}' "${CSV_FILE}")
- Thermal Zone 1: $(awk -F',' 'NR>1 {sum+=$12; count++} END {print sum/count "°C average"}' "${CSV_FILE}")
- Thermal Zone 2: $(awk -F',' 'NR>1 {sum+=$13; count++} END {print sum/count "°C average"}' "${CSV_FILE}")

Peak Values:
- Max CPU Frequency: $(awk -F',' 'NR>1 {if($2>max) max=$2} END {print max " MHz"}' "${CSV_FILE}")
- Max GPU Frequency: $(awk -F',' 'NR>1 {if($3>max) max=$3} END {print max " MHz"}' "${CSV_FILE}")
- Max RAM Usage: $(awk -F',' 'NR>1 {if($5>max) max=$5} END {print max " MB"}' "${CSV_FILE}")
- Max Temperature: $(awk -F',' 'NR>1 {if($11>max) max=$11} END {print max "°C"}' "${CSV_FILE}")
EOF
    
    echo -e "${GREEN}Summary report generated${NC}"
}

# Function to create visualization script
create_visualization_script() {
    echo -e "${YELLOW}Creating visualization script...${NC}"
    
    cat > "${OUTPUT_DIR}/plot_telemetry.py" << 'EOF'
#!/usr/bin/env python3
"""
PlowPilot AI-Vision Telemetry Visualization
Plot telemetry data from CSV file
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

def plot_telemetry(csv_file):
    """Plot telemetry data from CSV file"""
    
    # Read CSV data
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PlowPilot AI-Vision Telemetry', fontsize=16)
    
    # Plot 1: CPU and GPU frequencies
    axes[0, 0].plot(df['timestamp'], df['cpu_freq_mhz'], label='CPU Freq (MHz)', color='blue')
    axes[0, 0].plot(df['timestamp'], df['gpu_freq_mhz'], label='GPU Freq (MHz)', color='red')
    axes[0, 0].set_title('CPU and GPU Frequencies')
    axes[0, 0].set_ylabel('Frequency (MHz)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Memory usage
    axes[0, 1].plot(df['timestamp'], df['ram_usage_mb'], label='RAM Usage (MB)', color='green')
    axes[0, 1].plot(df['timestamp'], df['swap_usage_mb'], label='Swap Usage (MB)', color='orange')
    axes[0, 1].set_title('Memory Usage')
    axes[0, 1].set_ylabel('Usage (MB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: GPU and EMC utilization
    axes[1, 0].plot(df['timestamp'], df['gpu_util_percent'], label='GPU Util (%)', color='purple')
    axes[1, 0].plot(df['timestamp'], df['emc_util_percent'], label='EMC Util (%)', color='brown')
    axes[1, 0].set_title('GPU and EMC Utilization')
    axes[1, 0].set_ylabel('Utilization (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Temperature
    axes[1, 1].plot(df['timestamp'], df['thermal_zone0_temp'], label='Thermal Zone 0', color='red')
    axes[1, 1].plot(df['timestamp'], df['thermal_zone1_temp'], label='Thermal Zone 1', color='blue')
    axes[1, 1].plot(df['timestamp'], df['thermal_zone2_temp'], label='Thermal Zone 2', color='green')
    axes[1, 1].set_title('Temperature')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Rotate x-axis labels
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = csv_file.replace('.csv', '_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot_telemetry.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)
    
    plot_telemetry(csv_file)
EOF
    
    chmod +x "${OUTPUT_DIR}/plot_telemetry.py"
    echo -e "${GREEN}Visualization script created: ${OUTPUT_DIR}/plot_telemetry.py${NC}"
}

# Main execution
main() {
    echo "Starting telemetry benchmark..."
    echo "Duration: ${BENCHMARK_DURATION} seconds"
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""
    
    # Collect system info
    collect_system_info
    
    # Run benchmark
    run_benchmark
    
    # Parse data
    parse_tegrastats
    
    # Generate summary
    generate_summary
    
    # Create visualization script
    create_visualization_script
    
    echo -e "${GREEN}Benchmark completed successfully!${NC}"
    echo ""
    echo "Output files:"
    echo "  - Log: ${LOG_FILE}"
    echo "  - CSV: ${CSV_FILE}"
    echo "  - System info: ${OUTPUT_DIR}/system_info.txt"
    echo "  - Summary: ${OUTPUT_DIR}/summary_*.txt"
    echo "  - Plot script: ${OUTPUT_DIR}/plot_telemetry.py"
    echo ""
    echo "To visualize data:"
    echo "  python3 ${OUTPUT_DIR}/plot_telemetry.py ${CSV_FILE}"
}

# Run main function
main
