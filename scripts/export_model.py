#!/usr/bin/env python3
"""
PlowPilot AI-Vision Model Export Script
Export YOLOv8n to ONNX format for TensorRT conversion
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import onnx
import onnxruntime as ort

def load_config(config_file):
    """Load model configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def export_yolov8n_to_onnx(model_name="yolov8n", output_dir="models"):
    """Export YOLOv8n model to ONNX format"""
    print(f"Exporting {model_name} to ONNX format...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLOv8n model
    model = YOLO(f'{model_name}.pt')
    
    # Export to ONNX with proper settings for TensorRT
    onnx_path = os.path.join(output_dir, f'{model_name}.onnx')
    
    try:
        model.export(
            format='onnx',
            imgsz=640,
            optimize=True,
            simplify=True,
            opset=12,        # ✅ Updated to opset 12 for YOLOv8n compatibility
            dynamic=True,    # ✅ Enable dynamic shapes for flexibility
            batch=1,
            verbose=True     # ✅ Add verbose output for debugging
        )
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None
    
    # Move exported file to target location
    if os.path.exists(f'{model_name}.onnx'):
        os.rename(f'{model_name}.onnx', onnx_path)
    
    print(f"ONNX model exported to: {onnx_path}")
    return onnx_path

def validate_onnx_model(onnx_path):
    """Validate ONNX model"""
    print("Validating ONNX model...")
    
    try:
        # Load ONNX model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✓ ONNX model is valid")
        
        # Test inference with ONNX Runtime
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        print(f"✓ Input name: {input_name}")
        print(f"✓ Input shape: {input_shape}")
        
        # Test with dummy input
        import numpy as np
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"✓ Output shapes: {[output.shape for output in outputs]}")
        print("✓ ONNX model inference test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX model validation failed: {e}")
        return False

def create_calibration_data(output_dir="data/calibration", num_samples=100):
    """Create calibration data for INT8 quantization"""
    print(f"Creating calibration data ({num_samples} samples)...")
    
    import cv2
    import numpy as np
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate calibration images
    for i in range(num_samples):
        # Create random image with some structure
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Add some structured content
        cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(img, (400, 400), 50, (128, 128, 128), -1)
        cv2.rectangle(img, (300, 300), (400, 400), (64, 64, 64), -1)
        
        # Add text
        cv2.putText(img, f"Calib {i:03d}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save image
        filename = os.path.join(output_dir, f"calib_{i:03d}.jpg")
        cv2.imwrite(filename, img)
    
    print(f"Calibration data created: {num_samples} samples in {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8n model to ONNX format')
    parser.add_argument('--config', type=str, default='configs/model.yaml',
                       help='Model configuration file')
    parser.add_argument('--model', type=str, default='yolov8n',
                       help='Model name to export')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--calibration-data', type=str, default='data/calibration',
                       help='Directory for calibration data')
    parser.add_argument('--num-calibration-samples', type=int, default=100,
                       help='Number of calibration samples to generate')
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported ONNX model')
    parser.add_argument('--create-calibration', action='store_true',
                       help='Create calibration data for INT8 quantization')
    
    args = parser.parse_args()
    
    print("PlowPilot AI-Vision Model Export")
    print("================================")
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        print(f"Warning: Configuration file not found: {args.config}")
        config = {}
    
    # Export model to ONNX
    onnx_path = export_yolov8n_to_onnx(args.model, args.output_dir)
    
    # Validate model if requested
    if args.validate:
        if not validate_onnx_model(onnx_path):
            print("Model validation failed!")
            sys.exit(1)
    
    # Create calibration data if requested
    if args.create_calibration:
        create_calibration_data(args.calibration_data, args.num_calibration_samples)
    
    print("Model export completed successfully!")
    print(f"ONNX model: {onnx_path}")
    print(f"Output directory: {args.output_dir}")
    
    if args.create_calibration:
        print(f"Calibration data: {args.calibration_data}")

if __name__ == "__main__":
    main()
