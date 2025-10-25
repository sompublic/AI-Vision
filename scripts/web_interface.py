#!/usr/bin/env python3
"""
PlowPilot AI-Vision Web Interface
Simple web interface for monitoring and control
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import yaml
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'plowpilot-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
config = {}
status = {
    'running': False,
    'fps': 0,
    'cpu_usage': 0,
    'gpu_usage': 0,
    'memory_usage': 0,
    'temperature': 0,
    'detections': [],
    'last_update': None
}

def load_config():
    """Load configuration from YAML files"""
    global config
    
    config_files = {
        'camera': 'configs/camera.yaml',
        'model': 'configs/model.yaml',
        'pipeline': 'configs/pipeline.yaml'
    }
    
    config = {}
    for key, file_path in config_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config[key] = yaml.safe_load(f)
        else:
            config[key] = {}

def get_system_status():
    """Get system status information"""
    try:
        # Get CPU usage
        with open('/proc/loadavg', 'r') as f:
            load_avg = float(f.read().split()[0])
        
        # Get memory usage
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        # Parse memory info
        mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1])
        mem_available = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1])
        mem_used = mem_total - mem_available
        mem_usage_percent = (mem_used / mem_total) * 100
        
        # Get temperature (simplified)
        temp = 0
        for temp_file in ['/sys/class/thermal/thermal_zone0/temp', '/sys/class/thermal/thermal_zone1/temp']:
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    temp = max(temp, int(f.read().strip()) / 1000)
        
        return {
            'cpu_usage': load_avg * 100,
            'memory_usage': mem_usage_percent,
            'temperature': temp
        }
    except Exception as e:
        print(f"Error getting system status: {e}")
        return {
            'cpu_usage': 0,
            'memory_usage': 0,
            'temperature': 0
        }

def update_status():
    """Update status information"""
    global status
    
    while True:
        try:
            # Get system status
            system_status = get_system_status()
            
            # Update status
            status.update(system_status)
            status['last_update'] = datetime.now().isoformat()
            
            # Emit status update to web clients
            socketio.emit('status_update', status)
            
            time.sleep(1)  # Update every second
            
        except Exception as e:
            print(f"Error updating status: {e}")
            time.sleep(5)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Get current status"""
    return jsonify(status)

@app.route('/api/config')
def api_config():
    """Get configuration"""
    return jsonify(config)

@app.route('/api/recordings')
def api_recordings():
    """Get list of recordings"""
    recordings_dir = Path('recordings')
    if not recordings_dir.exists():
        return jsonify([])
    
    recordings = []
    for file_path in recordings_dir.glob('*.mp4'):
        stat = file_path.stat()
        recordings.append({
            'name': file_path.name,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    return jsonify(sorted(recordings, key=lambda x: x['created'], reverse=True))

@app.route('/api/recordings/<filename>')
def download_recording(filename):
    """Download recording file"""
    file_path = Path('recordings') / filename
    if file_path.exists():
        return send_file(str(file_path), as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/control', methods=['POST'])
def api_control():
    """Control PlowPilot service"""
    data = request.get_json()
    action = data.get('action')
    
    if action == 'start':
        # Start PlowPilot service
        os.system('sudo systemctl start plowpilot')
        status['running'] = True
    elif action == 'stop':
        # Stop PlowPilot service
        os.system('sudo systemctl stop plowpilot')
        status['running'] = False
    elif action == 'restart':
        # Restart PlowPilot service
        os.system('sudo systemctl restart plowpilot')
        status['running'] = True
    
    return jsonify({'status': 'success', 'action': action})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status_update', status)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

def create_templates():
    """Create HTML templates"""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create index.html template
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PlowPilot AI-Vision Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { display: flex; justify-content: space-between; align-items: center; }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-running { background-color: #27ae60; }
        .status-stopped { background-color: #e74c3c; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-value { font-weight: bold; color: #2c3e50; }
        .controls { display: flex; gap: 10px; }
        .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-primary { background: #3498db; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-success { background: #27ae60; color: white; }
        .chart-container { position: relative; height: 300px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>PlowPilot AI-Vision Dashboard</h1>
            <p>Real-time video analytics on NVIDIA Jetson Orin Nano</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>System Status</h3>
                <div class="status">
                    <span>
                        <span class="status-indicator" id="status-indicator"></span>
                        <span id="status-text">Unknown</span>
                    </span>
                </div>
                <div class="metric">
                    <span>FPS:</span>
                    <span class="metric-value" id="fps">0</span>
                </div>
                <div class="metric">
                    <span>CPU Usage:</span>
                    <span class="metric-value" id="cpu-usage">0%</span>
                </div>
                <div class="metric">
                    <span>Memory Usage:</span>
                    <span class="metric-value" id="memory-usage">0%</span>
                </div>
                <div class="metric">
                    <span>Temperature:</span>
                    <span class="metric-value" id="temperature">0°C</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Controls</h3>
                <div class="controls">
                    <button class="btn btn-success" onclick="control('start')">Start</button>
                    <button class="btn btn-danger" onclick="control('stop')">Stop</button>
                    <button class="btn btn-primary" onclick="control('restart')">Restart</button>
                </div>
            </div>
            
            <div class="card">
                <h3>Performance Chart</h3>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>Recent Detections</h3>
                <div id="detections-list">
                    <p>No detections yet</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        const ctx = document.getElementById('performanceChart').getContext('2d');
        
        let performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'FPS',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }, {
                    label: 'CPU Usage %',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        socket.on('status_update', function(data) {
            updateStatus(data);
            updateChart(data);
        });
        
        function updateStatus(data) {
            document.getElementById('status-indicator').className = 
                'status-indicator ' + (data.running ? 'status-running' : 'status-stopped');
            document.getElementById('status-text').textContent = 
                data.running ? 'Running' : 'Stopped';
            document.getElementById('fps').textContent = data.fps || 0;
            document.getElementById('cpu-usage').textContent = Math.round(data.cpu_usage || 0) + '%';
            document.getElementById('memory-usage').textContent = Math.round(data.memory_usage || 0) + '%';
            document.getElementById('temperature').textContent = Math.round(data.temperature || 0) + '°C';
        }
        
        function updateChart(data) {
            const now = new Date().toLocaleTimeString();
            performanceChart.data.labels.push(now);
            performanceChart.data.datasets[0].data.push(data.fps || 0);
            performanceChart.data.datasets[1].data.push(data.cpu_usage || 0);
            
            if (performanceChart.data.labels.length > 20) {
                performanceChart.data.labels.shift();
                performanceChart.data.datasets[0].data.shift();
                performanceChart.data.datasets[1].data.shift();
            }
            
            performanceChart.update();
        }
        
        function control(action) {
            fetch('/api/control', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({action: action})
            })
            .then(response => response.json())
            .then(data => {
                console.log('Control action:', data);
            });
        }
    </script>
</body>
</html>'''
    
    with open(templates_dir / 'index.html', 'w') as f:
        f.write(index_html)

def main():
    """Main function"""
    print("PlowPilot AI-Vision Web Interface")
    print("=================================")
    
    # Load configuration
    load_config()
    
    # Create templates
    create_templates()
    
    # Start status update thread
    status_thread = threading.Thread(target=update_status, daemon=True)
    status_thread.start()
    
    # Start web server
    port = int(os.environ.get('PLOWPILOT_WEB_PORT', 8080))
    print(f"Starting web interface on port {port}")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    main()
