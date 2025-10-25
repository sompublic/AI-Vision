#!/bin/bash
# run_jetson_clocks.sh - Set NVIDIA Jetson performance mode and clocks
# PlowPilot AI-Vision - NVIDIA Jetson Orin Nano

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}PlowPilot AI-Vision - Jetson Clock Configuration${NC}"
echo "=================================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${RED}Error: Not running on NVIDIA Jetson${NC}"
    exit 1
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Function to set max performance mode
set_max_performance() {
    echo -e "${YELLOW}Setting Jetson to maximum performance mode...${NC}"
    
    # Set CPU governor to performance
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    
    # Set GPU to max frequency
    echo 1 | tee /sys/devices/gpu.0/load
    
    # Set EMC to max frequency
    echo 1 | tee /sys/kernel/debug/emc/load
    
    # Set fan to max speed (if available)
    if [ -f /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1 ]; then
        echo 255 | tee /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1
    fi
    
    echo -e "${GREEN}Maximum performance mode set${NC}"
}

# Function to set balanced performance mode
set_balanced_performance() {
    echo -e "${YELLOW}Setting Jetson to balanced performance mode...${NC}"
    
    # Set CPU governor to ondemand
    echo ondemand | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    
    # Set GPU to balanced frequency
    echo 0.5 | tee /sys/devices/gpu.0/load
    
    # Set EMC to balanced frequency
    echo 0.5 | tee /sys/kernel/debug/emc/load
    
    # Set fan to balanced speed
    if [ -f /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1 ]; then
        echo 128 | tee /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1
    fi
    
    echo -e "${GREEN}Balanced performance mode set${NC}"
}

# Function to set power saving mode
set_power_saving() {
    echo -e "${YELLOW}Setting Jetson to power saving mode...${NC}"
    
    # Set CPU governor to powersave
    echo powersave | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    
    # Set GPU to minimum frequency
    echo 0.1 | tee /sys/devices/gpu.0/load
    
    # Set EMC to minimum frequency
    echo 0.1 | tee /sys/kernel/debug/emc/load
    
    # Set fan to minimum speed
    if [ -f /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1 ]; then
        echo 0 | tee /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1
    fi
    
    echo -e "${GREEN}Power saving mode set${NC}"
}

# Function to show current status
show_status() {
    echo -e "${BLUE}Current Jetson Status:${NC}"
    echo "===================="
    
    # CPU frequency
    echo "CPU Frequency:"
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | head -4
    
    # CPU governor
    echo "CPU Governor:"
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | head -4
    
    # GPU frequency
    if [ -f /sys/devices/gpu.0/load ]; then
        echo "GPU Load:"
        cat /sys/devices/gpu.0/load
    fi
    
    # Temperature
    if [ -f /sys/class/thermal/thermal_zone*/temp ]; then
        echo "Temperature:"
        for temp in /sys/class/thermal/thermal_zone*/temp; do
            if [ -f "$temp" ]; then
                temp_c=$(cat "$temp")
                temp_c=$((temp_c / 1000))
                echo "  Zone $(basename $(dirname $temp)): ${temp_c}°C"
            fi
        done
    fi
    
    # Memory usage
    echo "Memory Usage:"
    free -h
    
    # Fan speed
    if [ -f /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1 ]; then
        fan_speed=$(cat /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1)
        echo "Fan Speed: ${fan_speed}/255"
    fi
}

# Function to run tegrastats for monitoring
run_tegrastats() {
    echo -e "${YELLOW}Starting tegrastats monitoring...${NC}"
    echo "Press Ctrl+C to stop"
    
    if command -v tegrastats &> /dev/null; then
        tegrastats --interval 1000 --logfile tegrastats.log
    else
        echo -e "${RED}Error: tegrastats not found${NC}"
        exit 1
    fi
}

# Function to create systemd service for automatic clock setting
create_systemd_service() {
    echo -e "${YELLOW}Creating systemd service for automatic clock setting...${NC}"
    
    cat > /etc/systemd/system/plowpilot-clocks.service << EOF
[Unit]
Description=PlowPilot AI-Vision Clock Configuration
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/home/neo/AI-RESEARCH-LAB/AI-Vision/scripts/run_jetson_clocks.sh max
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable plowpilot-clocks.service
    
    echo -e "${GREEN}Systemd service created and enabled${NC}"
    echo "To start: sudo systemctl start plowpilot-clocks"
    echo "To stop: sudo systemctl stop plowpilot-clocks"
}

# Main execution
main() {
    case "${1:-status}" in
        "max"|"maximum")
            set_max_performance
            ;;
        "balanced"|"balance")
            set_balanced_performance
            ;;
        "power"|"powersave")
            set_power_saving
            ;;
        "status")
            show_status
            ;;
        "monitor"|"tegrastats")
            run_tegrastats
            ;;
        "install"|"service")
            create_systemd_service
            ;;
        *)
            echo -e "${YELLOW}Usage: $0 [max|balanced|power|status|monitor|install]${NC}"
            echo ""
            echo "Options:"
            echo "  max       - Set maximum performance mode"
            echo "  balanced  - Set balanced performance mode"
            echo "  power     - Set power saving mode"
            echo "  status    - Show current status"
            echo "  monitor   - Run tegrastats monitoring"
            echo "  install   - Create systemd service"
            echo ""
            echo "Examples:"
            echo "  $0 max      # Set maximum performance"
            echo "  $0 status   # Show current status"
            echo "  $0 monitor  # Monitor with tegrastats"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
