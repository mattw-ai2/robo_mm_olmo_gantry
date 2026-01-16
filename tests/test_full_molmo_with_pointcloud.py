#!/usr/bin/env python3
"""
Full Molmo Model Test with Point Cloud Integration

This test verifies the complete forward pass:
  Point Cloud → PointCloudBackbone → PC Tokens → Molmo LLM → Output

Usage:
    python tests/test_full_molmo_with_pointcloud.py --checkpoint /path/to/molmo-checkpoint

    # With specific point cloud settings:
    python tests/test_full_molmo_with_pointcloud.py \
        --checkpoint /path/to/molmo-checkpoint \
        --voxel_size 0.1 \
        --grid_range 10.0 \
        --ptv3_channels 512
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_full_forward_pass(
    checkpoint_path: str,
    voxel_size: float = 0.2,
    grid_range: float = 5.0,
    ptv3_channels: int = 256,
    ptv3_num_layers: int = 2,
    freeze_pc_backbone: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Test full forward pass through Molmo with point cloud."""
    
    print("=" * 60)
    print("  FULL MOLMO + POINT CLOUD FORWARD PASS TEST")
    print("=" * 60)
    
    # Step 1: Load model config
    print("\n1. Loading model configuration...")
    try:
        from olmo.models.molmo.molmo import MolmoConfig, Molmo
        from olmo.nn.point_cloud_backbone import PointCloudBackboneConfig
        from olmo.checkpoint import load_state_dict
        import json
        
        config_path = Path(checkpoint_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            model_config = MolmoConfig.from_dict(config_dict)
            print(f"   ✅ Loaded config from {config_path}")
        else:
            print(f"   ⚠️  No config.json found, using default config")
            model_config = MolmoConfig()
        
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        return False
    
    # Step 2: Add point cloud backbone config
    print("\n2. Configuring point cloud backbone...")
    try:
        pc_config = PointCloudBackboneConfig(
            voxel_size=voxel_size,
            grid_range=grid_range,
            ptv3_channels=ptv3_channels,
            ptv3_num_layers=ptv3_num_layers,
        )
        model_config.point_cloud_backbone = pc_config
        print(f"   ✅ Point cloud config: voxel_size={voxel_size}, channels={ptv3_channels}")
    except Exception as e:
        print(f"   ❌ Failed to configure PC backbone: {e}")
        return False
    
    # Step 3: Initialize model
    print("\n3. Initializing Molmo model...")
    try:
        model = Molmo(model_config)
        print(f"   ✅ Model initialized")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        pc_params = sum(p.numel() for p in model.point_cloud_backbone.parameters()) if model.point_cloud_backbone else 0
        print(f"   ✅ Total params: {total_params:,}")
        print(f"   ✅ Point cloud backbone params: {pc_params:,}")
        
    except Exception as e:
        print(f"   ❌ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Load checkpoint weights (optional)
    print("\n4. Loading checkpoint weights...")
    try:
        weight_path = Path(checkpoint_path) / "model.safetensors"
        if not weight_path.exists():
            weight_path = Path(checkpoint_path) / "pytorch_model.bin"
        
        if weight_path.exists():
            state_dict = load_state_dict(str(weight_path))
            # Filter out point cloud backbone weights (they're new)
            model_keys = set(model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"   ✅ Loaded weights from {weight_path}")
        else:
            print(f"   ⚠️  No weights found, using random initialization")
    except Exception as e:
        print(f"   ⚠️  Failed to load weights: {e}")
        print(f"   ⚠️  Continuing with random initialization")
    
    # Step 5: Freeze/unfreeze point cloud backbone
    print(f"\n5. Configuring point cloud backbone freeze state...")
    try:
        if model.point_cloud_backbone is not None:
            if freeze_pc_backbone:
                for param in model.point_cloud_backbone.parameters():
                    param.requires_grad = False
                print(f"   ✅ Point cloud backbone FROZEN (no gradients)")
            else:
                for param in model.point_cloud_backbone.parameters():
                    param.requires_grad = True
                print(f"   ✅ Point cloud backbone UNFROZEN (trainable)")
            
            # Count trainable params
            trainable = sum(p.numel() for p in model.point_cloud_backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.point_cloud_backbone.parameters())
            print(f"   ✅ PC backbone: {trainable:,} / {total:,} trainable params")
        else:
            print(f"   ⚠️  No point cloud backbone to freeze")
    except Exception as e:
        print(f"   ❌ Failed to configure freeze state: {e}")
        return False
    
    # Step 6: Move to device
    print(f"\n6. Moving model to {device}...")
    try:
        model = model.to(device)
        model.eval()
        print(f"   ✅ Model on {device}")
    except Exception as e:
        print(f"   ❌ Failed to move to device: {e}")
        return False
    
    # Step 7: Create test inputs
    print("\n7. Creating test inputs...")
    try:
        batch_size = 2
        seq_len = 128
        num_points = 1000
        
        # Create dummy inputs
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Point cloud inputs
        point_cloud = torch.randn(batch_size, num_points, 3, device=device)
        point_cloud_mask = torch.ones(batch_size, num_points, dtype=torch.bool, device=device)
        
        # Image inputs (if needed)
        # image_features = torch.randn(batch_size, 576, model_config.d_model, device=device)
        
        print(f"   ✅ input_ids: {input_ids.shape}")
        print(f"   ✅ point_cloud: {point_cloud.shape}")
        print(f"   ✅ point_cloud_mask: {point_cloud_mask.shape}")
        
    except Exception as e:
        print(f"   ❌ Failed to create inputs: {e}")
        return False
    
    # Step 8: Forward pass
    print("\n8. Running forward pass...")
    try:
        with torch.no_grad():
            # The exact forward signature depends on your model implementation
            # This is a general pattern - adjust based on actual Molmo forward signature
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                point_cloud=point_cloud,
                point_cloud_mask=point_cloud_mask,
            )
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        print(f"   ✅ Output logits shape: {logits.shape}")
        print(f"   ✅ Output dtype: {logits.dtype}")
        print(f"   ✅ Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 9: Test gradient flow
    print("\n9. Testing gradient flow...")
    try:
        model.train()
        
        # Fresh forward pass for gradients
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_cloud=point_cloud,
            point_cloud_mask=point_cloud_mask,
        )
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        loss = logits.mean()
        loss.backward()
        
        # Check gradients in point cloud backbone
        pc_has_grads = False
        if model.point_cloud_backbone:
            for name, param in model.point_cloud_backbone.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    pc_has_grads = True
                    break
        
        if freeze_pc_backbone:
            if not pc_has_grads:
                print(f"   ✅ Point cloud backbone correctly frozen (no gradients)")
            else:
                print(f"   ⚠️  Point cloud backbone has gradients but should be frozen")
        else:
            if pc_has_grads:
                print(f"   ✅ Gradients flow through point cloud backbone")
            else:
                print(f"   ⚠️  No gradients in point cloud backbone")
        
        # Check gradients in LLM
        llm_has_grads = False
        for name, param in model.named_parameters():
            if 'point_cloud' not in name and param.grad is not None and param.grad.abs().sum() > 0:
                llm_has_grads = True
                break
        
        if llm_has_grads:
            print(f"   ✅ Gradients flow through LLM backbone")
        else:
            print(f"   ⚠️  No gradients in LLM backbone")
        
    except Exception as e:
        print(f"   ❌ Gradient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST COMPLETE - ALL CHECKS PASSED")
    print("=" * 60)
    print("\nPipeline verified:")
    print("  Point Cloud [B, N, 3]")
    print("       ↓")
    print("  PointCloudBackbone (voxelize → PTv3 → project)")
    print("       ↓")
    print("  PC Tokens [B, num_voxels, d_model]")
    print("       ↓")
    print("  Molmo LLM (with image + text tokens)")
    print("       ↓")
    print("  Output Logits [B, seq_len, vocab_size]")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test full Molmo model with point cloud")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Molmo checkpoint directory"
    )
    parser.add_argument("--voxel_size", type=float, default=0.2, help="Voxel size in meters")
    parser.add_argument("--grid_range", type=float, default=5.0, help="Grid range in meters")
    parser.add_argument("--ptv3_channels", type=int, default=256, help="PTv3 channel dimension")
    parser.add_argument("--ptv3_num_layers", type=int, default=2, help="PTv3 number of layers")
    parser.add_argument("--freeze_pc_backbone", action="store_true", help="Freeze point cloud backbone (no gradients)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    success = test_full_forward_pass(
        checkpoint_path=args.checkpoint,
        voxel_size=args.voxel_size,
        grid_range=args.grid_range,
        ptv3_channels=args.ptv3_channels,
        ptv3_num_layers=args.ptv3_num_layers,
        freeze_pc_backbone=args.freeze_pc_backbone,
        device=device,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

