#!/usr/bin/env python
"""
Quick test script for VGGT point cloud generation.

Usage:
    # Activate vggt environment first
    conda activate vggt
    
    # Run test
    python scripts/test_vggt.py
"""

import os
import sys
import numpy as np

print("=" * 60)
print("VGGT Point Cloud Generation Test")
print("=" * 60)

# Test 1: Check VGGT import
print("\n1. Testing VGGT import...")
try:
    from vggt.models.vggt import VGGT
    print("   ✓ VGGT imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import VGGT: {e}")
    print("   Install with: pip install git+https://github.com/facebookresearch/vggt.git")
    sys.exit(1)

# Test 2: Check torch/CUDA
print("\n2. Checking PyTorch and CUDA...")
import torch
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("   ⚠ No CUDA, using CPU (will be slow)")
    device = "cpu"

# Test 3: Load VGGT model
print("\n3. Loading VGGT model (this may take a minute)...")
try:
    model = VGGT.from_pretrained("facebook/vggt-1b")
    model = model.to(device)
    model.eval()
    print("   ✓ VGGT model loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    sys.exit(1)

# Test 4: Test on synthetic images
print("\n4. Testing inference on synthetic images...")
try:
    # Create dummy images (3 frames, RGB, 224x224)
    dummy_images = torch.randn(1, 3, 3, 224, 224).to(device)  # [B, T, C, H, W]
    
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"   ✓ Inference successful!")
    print(f"   Output keys: {list(outputs.keys())}")
    
    if "depth" in outputs:
        print(f"   Depth shape: {outputs['depth'].shape}")
    if "world_points" in outputs:
        print(f"   World points shape: {outputs['world_points'].shape}")
        
except Exception as e:
    print(f"   ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test on real VIDA data (if available)
print("\n5. Testing on real VIDA data...")
VIDA_PATH = "/weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor/ObjectNavType/train/000000"

if os.path.exists(VIDA_PATH):
    try:
        # Try to load a video frame
        import cv2
        video_path = os.path.join(VIDA_PATH, "raw_front_camera__0.mp4")
        
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"   ✓ Loaded frame from {video_path}")
                print(f"   Frame shape: {frame.shape}")
                
                # Convert to tensor and run inference
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, C, H, W]
                
                with torch.no_grad():
                    outputs = model(frame_tensor)
                
                print(f"   ✓ Real image inference successful!")
                if "depth" in outputs:
                    depth = outputs["depth"].cpu().numpy()
                    print(f"   Depth range: {depth.min():.2f} - {depth.max():.2f}")
            else:
                print(f"   ⚠ Could not read frame from video")
        else:
            print(f"   ⚠ Video not found: {video_path}")
    except Exception as e:
        print(f"   ⚠ Error testing real data: {e}")
else:
    print(f"   ⚠ VIDA data not found at {VIDA_PATH}")

print("\n" + "=" * 60)
print("VGGT Test Complete!")
print("=" * 60)

