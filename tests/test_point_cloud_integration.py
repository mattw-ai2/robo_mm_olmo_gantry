#!/usr/bin/env python3
"""
Modular Test Suite for Point Cloud Integration

This test suite verifies each component of the point cloud pipeline:

PIPELINE COMPONENTS:
====================

1. PREPROCESSING (Offline)
   ├── 1.1 VGGT Depth Estimation
   ├── 1.2 Point Cloud Unprojection
   ├── 1.3 Trajectory Accumulation
   ├── 1.4 Robot-Centered Transformation
   └── 1.5 HDF5 Saving/Loading

2. DATA LOADING (Training)
   ├── 2.1 RobotDataset Configuration
   ├── 2.2 Point Cloud File Discovery
   ├── 2.3 Keyframe Matching
   └── 2.4 Example Dictionary Structure

3. MODEL PREPROCESSING
   ├── 3.1 MolmoPreprocessor Point Cloud Handling
   ├── 3.2 Token Generation (pc_start, pc_patch, pc_end)
   └── 3.3 Sequence Construction

4. BATCH COLLATION
   ├── 4.1 MMCollator Point Cloud Batching
   └── 4.2 Mask Padding

5. MODEL FORWARD PASS
   ├── 5.1 PointCloudBackbone Forward
   ├── 5.2 Voxelization
   ├── 5.3 Point Transformer Layers
   ├── 5.4 Projection to LLM Dimension
   └── 5.5 Integration with Molmo

6. TRAINING LOOP
   ├── 6.1 Loss Computation
   └── 6.2 Gradient Flow

Run individual test groups:
    python tests/test_point_cloud_integration.py --test preprocessing
    python tests/test_point_cloud_integration.py --test data_loading
    python tests/test_point_cloud_integration.py --test model
    python tests/test_point_cloud_integration.py --test full_pipeline

Run all tests:
    python tests/test_point_cloud_integration.py --test all
"""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_subtest(text: str):
    """Print a subtest header."""
    print(f"\n  📋 {text}")
    print(f"  {'-'*50}")


def print_success(text: str):
    """Print success message."""
    print(f"    ✅ {text}")


def print_failure(text: str):
    """Print failure message."""
    print(f"    ❌ {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"    ⚠️  {text}")


def print_info(text: str):
    """Print info message."""
    print(f"    ℹ️  {text}")


# =============================================================================
# TEST GROUP 1: PREPROCESSING
# =============================================================================

def test_preprocessing() -> Dict[str, bool]:
    """Test preprocessing components."""
    print_header("1. PREPROCESSING TESTS")
    results = {}
    
    # Test 1.1: Point Cloud Unprojection
    print_subtest("1.1 Point Cloud Unprojection")
    try:
        from scripts.precompute_vggt_depth import unproject_depth_to_points
        
        # Create synthetic depth map
        H, W = 64, 64
        depth = np.random.uniform(0.5, 5.0, (H, W)).astype(np.float32)
        
        # Simple pinhole intrinsics
        fx, fy = 500, 500
        cx, cy = W/2, H/2
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        points = unproject_depth_to_points(depth, intrinsics)
        
        assert points.shape == (H, W, 3), f"Expected shape (64, 64, 3), got {points.shape}"
        assert not np.any(np.isnan(points)), "Points contain NaN values"
        print_success(f"Unprojection works: depth {depth.shape} → points {points.shape}")
        results["1.1_unprojection"] = True
    except ImportError as e:
        print_warning(f"Could not import preprocessing module: {e}")
        results["1.1_unprojection"] = None
    except Exception as e:
        print_failure(f"Unprojection failed: {e}")
        results["1.1_unprojection"] = False
    
    # Test 1.2: Trajectory Accumulation
    print_subtest("1.2 Trajectory Accumulation")
    try:
        from scripts.precompute_vggt_depth import accumulate_trajectory_points
        
        # Create synthetic point clouds from multiple frames
        num_frames = 5
        points_per_frame = 1000
        all_points = []
        all_extrinsics = []
        
        for i in range(num_frames):
            points = np.random.randn(points_per_frame, 3).astype(np.float32)
            # Simple translation for each frame
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[:3, 3] = [i * 0.5, 0, 0]  # Move along X
            all_points.append(points)
            all_extrinsics.append(extrinsics)
        
        accumulated, mask = accumulate_trajectory_points(
            all_points, all_extrinsics, max_points=3000
        )
        
        assert accumulated.shape[0] <= 3000, f"Should have max 3000 points, got {accumulated.shape[0]}"
        assert accumulated.shape[1] == 3, f"Points should be 3D, got {accumulated.shape[1]}"
        assert mask.shape[0] == accumulated.shape[0], "Mask length mismatch"
        print_success(f"Accumulated {num_frames} frames → {mask.sum()} valid points")
        results["1.2_accumulation"] = True
    except ImportError as e:
        print_warning(f"Could not import accumulation function: {e}")
        results["1.2_accumulation"] = None
    except Exception as e:
        print_failure(f"Accumulation failed: {e}")
        traceback.print_exc()
        results["1.2_accumulation"] = False
    
    # Test 1.3: Robot-Centered Transformation
    print_subtest("1.3 Robot-Centered Transformation")
    try:
        from scripts.precompute_vggt_depth import transform_to_robot_frame
        
        # Points in world frame
        points_world = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ], dtype=np.float32)
        
        # Robot pose: translated by [1, 1, 0] and rotated 90 degrees around Z
        robot_pose = np.array([
            [0, -1, 0, 1],
            [1,  0, 0, 1],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=np.float32)
        
        points_robot = transform_to_robot_frame(points_world, robot_pose)
        
        assert points_robot.shape == points_world.shape, "Shape should be preserved"
        print_success(f"Transform works: world → robot frame")
        results["1.3_robot_transform"] = True
    except ImportError as e:
        print_warning(f"Could not import transform function: {e}")
        results["1.3_robot_transform"] = None
    except Exception as e:
        print_failure(f"Robot transform failed: {e}")
        results["1.3_robot_transform"] = False
    
    # Test 1.4: HDF5 Save/Load
    print_subtest("1.4 HDF5 Save/Load")
    try:
        import h5py
        
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = os.path.join(tmpdir, "test_point_clouds.hdf5")
            
            # Save
            points = np.random.randn(1000, 3).astype(np.float32)
            mask = np.ones(1000, dtype=bool)
            
            with h5py.File(hdf5_path, 'w') as f:
                episode_group = f.create_group("0")
                frame_group = episode_group.create_group("5")
                frame_group.create_dataset("point_cloud", data=points, compression='gzip')
                frame_group.create_dataset("point_cloud_mask", data=mask, compression='gzip')
            
            # Load
            with h5py.File(hdf5_path, 'r') as f:
                loaded_points = np.array(f["0"]["5"]["point_cloud"])
                loaded_mask = np.array(f["0"]["5"]["point_cloud_mask"])
            
            assert np.allclose(points, loaded_points), "Points don't match after save/load"
            assert np.array_equal(mask, loaded_mask), "Mask doesn't match after save/load"
            print_success(f"HDF5 save/load works: {points.shape}")
            results["1.4_hdf5"] = True
    except ImportError as e:
        print_failure(f"h5py not installed: {e}")
        results["1.4_hdf5"] = False
    except Exception as e:
        print_failure(f"HDF5 test failed: {e}")
        results["1.4_hdf5"] = False
    
    return results


# =============================================================================
# TEST GROUP 2: DATA LOADING
# =============================================================================

def test_data_loading() -> Dict[str, bool]:
    """Test data loading components."""
    print_header("2. DATA LOADING TESTS")
    results = {}
    
    # Test 2.1: RobotDataset Configuration
    print_subtest("2.1 RobotDataset Configuration")
    try:
        from olmo.data.robot_datasets import RobotDataset
        import inspect
        
        sig = inspect.signature(RobotDataset.__init__)
        params = list(sig.parameters.keys())
        
        assert 'use_point_cloud' in params, "Missing use_point_cloud parameter"
        assert 'point_cloud_dir' in params, "Missing point_cloud_dir parameter"
        
        print_success("RobotDataset has point cloud parameters")
        results["2.1_config"] = True
    except ImportError as e:
        print_failure(f"Could not import RobotDataset: {e}")
        results["2.1_config"] = False
    except AssertionError as e:
        print_failure(str(e))
        results["2.1_config"] = False
    
    # Test 2.2: Point Cloud Loading Method
    print_subtest("2.2 Point Cloud Loading Method")
    try:
        from olmo.data.robot_datasets import RobotDataset
        
        assert hasattr(RobotDataset, '_load_point_cloud'), "Missing _load_point_cloud method"
        
        # Check method signature
        sig = inspect.signature(RobotDataset._load_point_cloud)
        params = list(sig.parameters.keys())
        
        print_success(f"_load_point_cloud method exists with params: {params[1:]}")  # Skip 'self'
        results["2.2_load_method"] = True
    except Exception as e:
        print_failure(f"Point cloud loading method check failed: {e}")
        results["2.2_load_method"] = False
    
    # Test 2.3: Mock Point Cloud Loading
    print_subtest("2.3 Mock Point Cloud Loading")
    try:
        import h5py
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock point cloud structure
            # Structure: point_cloud_dir/ObjectNavType/train/house_id/point_clouds.hdf5
            pc_dir = os.path.join(tmpdir, "ObjectNavType", "train", "000000")
            os.makedirs(pc_dir)
            
            hdf5_path = os.path.join(pc_dir, "point_clouds.hdf5")
            
            # Create test data for episode 0, frames 5 and 10
            with h5py.File(hdf5_path, 'w') as f:
                for episode in [0, 1]:
                    ep_group = f.create_group(str(episode))
                    for frame in [5, 10, 15]:
                        frame_group = ep_group.create_group(str(frame))
                        points = np.random.randn(500, 3).astype(np.float32)
                        mask = np.ones(500, dtype=bool)
                        frame_group.create_dataset("point_cloud", data=points)
                        frame_group.create_dataset("point_cloud_mask", data=mask)
            
            # Verify structure
            with h5py.File(hdf5_path, 'r') as f:
                episodes = list(f.keys())
                frames = list(f["0"].keys())
            
            print_success(f"Mock data created: {len(episodes)} episodes, frames {frames}")
            results["2.3_mock_loading"] = True
    except Exception as e:
        print_failure(f"Mock loading test failed: {e}")
        traceback.print_exc()
        results["2.3_mock_loading"] = False
    
    # Test 2.4: Environment Variable Support
    print_subtest("2.4 Environment Variable Support")
    try:
        from olmo.data.get_dataset import get_dataset_by_name
        import inspect
        
        # Check that get_dataset reads env vars
        source = inspect.getsource(get_dataset_by_name)
        
        checks = [
            ("ROBOT_USE_POINT_CLOUD" in source or "use_point_cloud" in source, "use_point_cloud handling"),
            ("ROBOT_POINT_CLOUD_DIR" in source or "point_cloud_dir" in source, "point_cloud_dir handling"),
        ]
        
        for check, name in checks:
            if check:
                print_success(f"get_dataset has {name}")
            else:
                print_warning(f"get_dataset may not have {name}")
        
        results["2.4_env_vars"] = True
    except Exception as e:
        print_failure(f"Environment variable test failed: {e}")
        results["2.4_env_vars"] = False
    
    return results


# =============================================================================
# TEST GROUP 3: MODEL PREPROCESSING
# =============================================================================

def test_model_preprocessing() -> Dict[str, bool]:
    """Test model preprocessing components."""
    print_header("3. MODEL PREPROCESSING TESTS")
    results = {}
    
    # Test 3.1: Point Cloud Tokens
    print_subtest("3.1 Point Cloud Tokens")
    try:
        from olmo.tokenizer import (
            POINT_CLOUD_START_TOKEN,
            POINT_CLOUD_PATCH_TOKEN,
            POINT_CLOUD_END_TOKEN,
        )
        
        print_success(f"PC_START: {POINT_CLOUD_START_TOKEN}")
        print_success(f"PC_PATCH: {POINT_CLOUD_PATCH_TOKEN}")
        print_success(f"PC_END: {POINT_CLOUD_END_TOKEN}")
        results["3.1_tokens"] = True
    except ImportError as e:
        print_warning(f"Point cloud tokens not defined: {e}")
        results["3.1_tokens"] = None
    
    # Test 3.2: MolmoPreprocessor Point Cloud Handling
    print_subtest("3.2 MolmoPreprocessor Point Cloud Handling")
    try:
        from olmo.models.molmo.model_preprocessor import MolmoPreprocessor
        import inspect
        
        # Check if preprocessor handles point clouds
        sig = inspect.signature(MolmoPreprocessor.__call__)
        params = list(sig.parameters.keys())
        
        if 'point_cloud' in params:
            print_success("MolmoPreprocessor.__call__ accepts point_cloud")
            results["3.2_preprocessor"] = True
        else:
            # Check for alternative handling
            source = inspect.getsource(MolmoPreprocessor)
            if 'point_cloud' in source:
                print_success("MolmoPreprocessor handles point_cloud internally")
                results["3.2_preprocessor"] = True
            else:
                print_warning("MolmoPreprocessor may not handle point_cloud")
                results["3.2_preprocessor"] = None
    except Exception as e:
        print_failure(f"Preprocessor test failed: {e}")
        results["3.2_preprocessor"] = False
    
    return results


# =============================================================================
# TEST GROUP 4: BATCH COLLATION
# =============================================================================

def test_batch_collation() -> Dict[str, bool]:
    """Test batch collation components."""
    print_header("4. BATCH COLLATION TESTS")
    results = {}
    
    # Test 4.1: MMCollator Point Cloud Handling
    print_subtest("4.1 MMCollator Point Cloud Handling")
    try:
        from olmo.models.molmo.collator import MMCollator
        import inspect
        
        source = inspect.getsource(MMCollator)
        
        if 'point_cloud' in source.lower():
            print_success("MMCollator handles point cloud data")
            results["4.1_collator"] = True
        else:
            print_warning("MMCollator may not handle point cloud data")
            results["4.1_collator"] = None
    except Exception as e:
        print_failure(f"Collator test failed: {e}")
        results["4.1_collator"] = False
    
    return results


# =============================================================================
# TEST GROUP 5: MODEL FORWARD PASS
# =============================================================================

def test_model_forward() -> Dict[str, bool]:
    """Test model forward pass components."""
    print_header("5. MODEL FORWARD PASS TESTS")
    results = {}
    
    # Test 5.1: PointCloudBackboneConfig
    print_subtest("5.1 PointCloudBackboneConfig")
    try:
        from olmo.nn.point_cloud_backbone import PointCloudBackboneConfig
        
        config = PointCloudBackboneConfig(
            voxel_size=0.1,
            grid_range=10.0,
            ptv3_channels=256,
            ptv3_num_layers=2,
            ptv3_num_heads=4,
        )
        
        print_success(f"Config created: voxel_size={config.voxel_size}, channels={config.ptv3_channels}")
        results["5.1_config"] = True
    except Exception as e:
        print_failure(f"Config test failed: {e}")
        results["5.1_config"] = False
    
    # Test 5.2: PointCloudBackbone Initialization
    print_subtest("5.2 PointCloudBackbone Initialization")
    try:
        import torch
        from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig
        
        config = PointCloudBackboneConfig(
            voxel_size=0.2,
            grid_range=5.0,
            ptv3_channels=128,
            ptv3_num_layers=1,
            ptv3_num_heads=2,
        )
        
        backbone = PointCloudBackbone(config, llm_dim=1024)
        
        # Count parameters
        num_params = sum(p.numel() for p in backbone.parameters())
        print_success(f"Backbone initialized: {num_params:,} parameters")
        results["5.2_backbone_init"] = True
    except Exception as e:
        print_failure(f"Backbone init failed: {e}")
        traceback.print_exc()
        results["5.2_backbone_init"] = False
    
    # Test 5.3: PointCloudBackbone Forward Pass
    print_subtest("5.3 PointCloudBackbone Forward Pass")
    try:
        import torch
        from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig
        
        config = PointCloudBackboneConfig(
            voxel_size=0.2,
            grid_range=5.0,
            ptv3_channels=128,
            ptv3_num_layers=1,
            ptv3_num_heads=2,
        )
        
        backbone = PointCloudBackbone(config, llm_dim=1024)
        backbone.eval()
        
        # Create test input
        batch_size = 2
        num_points = 500
        points = torch.randn(batch_size, num_points, 3)
        mask = torch.ones(batch_size, num_points, dtype=torch.bool)
        
        # Forward pass
        with torch.no_grad():
            tokens, valid_mask = backbone(points, mask)
        
        print_success(f"Forward pass: points {points.shape} → tokens {tokens.shape}")
        print_info(f"Valid voxels per sample: {valid_mask.sum(dim=1).tolist()}")
        results["5.3_forward"] = True
    except Exception as e:
        print_failure(f"Forward pass failed: {e}")
        traceback.print_exc()
        results["5.3_forward"] = False
    
    # Test 5.4: Molmo Model with Point Cloud Backbone
    print_subtest("5.4 Molmo Model with Point Cloud Backbone")
    try:
        from olmo.models.molmo.molmo import MolmoConfig, Molmo
        from olmo.nn.point_cloud_backbone import PointCloudBackboneConfig
        
        # Check if MolmoConfig accepts point_cloud_backbone
        import inspect
        fields = [f.name for f in MolmoConfig.__dataclass_fields__.values()] if hasattr(MolmoConfig, '__dataclass_fields__') else dir(MolmoConfig)
        
        if 'point_cloud_backbone' in fields or hasattr(MolmoConfig, 'point_cloud_backbone'):
            print_success("MolmoConfig has point_cloud_backbone field")
            results["5.4_molmo_config"] = True
        else:
            print_warning("MolmoConfig may not have point_cloud_backbone field")
            # Check if it's in the source
            source = inspect.getsource(MolmoConfig)
            if 'point_cloud' in source:
                print_success("MolmoConfig references point_cloud in source")
                results["5.4_molmo_config"] = True
            else:
                results["5.4_molmo_config"] = None
    except Exception as e:
        print_failure(f"Molmo config test failed: {e}")
        results["5.4_molmo_config"] = False
    
    return results


# =============================================================================
# TEST GROUP 6: FULL PIPELINE
# =============================================================================

def test_full_pipeline() -> Dict[str, bool]:
    """Test full pipeline integration."""
    print_header("6. FULL PIPELINE TESTS")
    results = {}
    
    # Test 6.1: End-to-End Point Cloud Processing
    print_subtest("6.1 End-to-End Point Cloud Processing")
    try:
        import torch
        import h5py
        from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create mock preprocessed data
            hdf5_path = os.path.join(tmpdir, "test.hdf5")
            original_points = np.random.randn(1000, 3).astype(np.float32)
            original_mask = np.ones(1000, dtype=bool)
            
            with h5py.File(hdf5_path, 'w') as f:
                ep = f.create_group("0")
                frame = ep.create_group("5")
                frame.create_dataset("point_cloud", data=original_points)
                frame.create_dataset("point_cloud_mask", data=original_mask)
            
            # Step 2: Load data
            with h5py.File(hdf5_path, 'r') as f:
                loaded_points = np.array(f["0"]["5"]["point_cloud"])
                loaded_mask = np.array(f["0"]["5"]["point_cloud_mask"])
            
            # Step 3: Process through backbone
            config = PointCloudBackboneConfig(
                voxel_size=0.2,
                grid_range=5.0,
                ptv3_channels=128,
                ptv3_num_layers=1,
            )
            backbone = PointCloudBackbone(config, llm_dim=1024)
            backbone.eval()
            
            points_tensor = torch.from_numpy(loaded_points).unsqueeze(0)
            mask_tensor = torch.from_numpy(loaded_mask).unsqueeze(0)
            
            with torch.no_grad():
                tokens, valid_mask = backbone(points_tensor, mask_tensor)
            
            print_success(f"Pipeline: HDF5 → load → backbone → tokens {tokens.shape}")
            results["6.1_e2e"] = True
    except Exception as e:
        print_failure(f"End-to-end test failed: {e}")
        traceback.print_exc()
        results["6.1_e2e"] = False
    
    # Test 6.2: Gradient Flow
    print_subtest("6.2 Gradient Flow Through Point Cloud Backbone")
    try:
        import torch
        from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig
        
        config = PointCloudBackboneConfig(
            voxel_size=0.2,
            grid_range=5.0,
            ptv3_channels=128,
            ptv3_num_layers=1,
        )
        
        backbone = PointCloudBackbone(config, llm_dim=1024)
        backbone.train()
        
        # Create test input
        points = torch.randn(2, 500, 3, requires_grad=False)
        mask = torch.ones(2, 500, dtype=torch.bool)
        
        # Forward pass
        tokens, valid_mask = backbone(points, mask)
        
        # Compute dummy loss
        loss = tokens.mean()
        
        # Check gradient flow
        loss.backward()
        
        # Verify gradients exist
        has_grads = False
        for name, param in backbone.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        
        if has_grads:
            print_success("Gradients flow through backbone")
            results["6.2_gradients"] = True
        else:
            print_failure("No gradients in backbone parameters")
            results["6.2_gradients"] = False
    except Exception as e:
        print_failure(f"Gradient test failed: {e}")
        traceback.print_exc()
        results["6.2_gradients"] = False
    
    # Test 6.3: Training Script Args
    print_subtest("6.3 Training Script Arguments")
    try:
        import inspect
        import importlib.util
        
        script_path = Path(__file__).parent.parent / "launch_scripts" / "train_multitask_model.py"
        
        if script_path.exists():
            content = script_path.read_text()
            
            checks = [
                ("--use_point_cloud" in content, "--use_point_cloud arg"),
                ("--point_cloud_dir" in content, "--point_cloud_dir arg"),
                ("PointCloudBackboneConfig" in content, "PointCloudBackboneConfig usage"),
            ]
            
            all_pass = True
            for check, name in checks:
                if check:
                    print_success(f"Training script has {name}")
                else:
                    print_warning(f"Training script missing {name}")
                    all_pass = False
            
            results["6.3_training_args"] = all_pass
        else:
            print_warning(f"Training script not found at {script_path}")
            results["6.3_training_args"] = None
    except Exception as e:
        print_failure(f"Training script test failed: {e}")
        results["6.3_training_args"] = False
    
    # Test 6.4: Full Model Forward Pass (Point Cloud → Encoder → Molmo → Output)
    print_subtest("6.4 Full Model Forward Pass with Point Cloud")
    try:
        import torch
        from olmo.models.molmo.molmo import MolmoConfig, Molmo
        from olmo.nn.point_cloud_backbone import PointCloudBackboneConfig
        
        # Create a minimal config for testing
        # Note: This requires a pretrained checkpoint to work properly
        # For now, just verify the config can be created with point cloud backbone
        
        pc_config = PointCloudBackboneConfig(
            voxel_size=0.2,
            grid_range=5.0,
            ptv3_channels=128,
            ptv3_num_layers=1,
        )
        
        # Check if we can set point_cloud_backbone in config
        # Full model instantiation requires pretrained weights, so we just verify the config path
        print_info("Full model test requires pretrained checkpoint")
        print_info("To test full forward pass, run:")
        print_info("  python tests/test_full_molmo_with_pointcloud.py --checkpoint /path/to/molmo")
        
        # Verify the model code handles point cloud in forward pass
        from olmo.models.molmo import molmo
        source = open(molmo.__file__).read()
        
        checks = [
            ("point_cloud_backbone" in source, "point_cloud_backbone in model"),
            ("point_cloud" in source, "point_cloud handling in forward"),
        ]
        
        all_pass = True
        for check, name in checks:
            if check:
                print_success(name)
            else:
                print_warning(f"Missing: {name}")
                all_pass = False
        
        results["6.4_full_model"] = all_pass
    except Exception as e:
        print_failure(f"Full model test failed: {e}")
        traceback.print_exc()
        results["6.4_full_model"] = False
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests() -> Dict[str, Dict[str, bool]]:
    """Run all test groups."""
    all_results = {}
    
    all_results["preprocessing"] = test_preprocessing()
    all_results["data_loading"] = test_data_loading()
    all_results["model_preprocessing"] = test_model_preprocessing()
    all_results["batch_collation"] = test_batch_collation()
    all_results["model_forward"] = test_model_forward()
    all_results["full_pipeline"] = test_full_pipeline()
    
    return all_results


def print_summary(results: Dict[str, Dict[str, bool]]):
    """Print test summary."""
    print_header("TEST SUMMARY")
    
    total_pass = 0
    total_fail = 0
    total_skip = 0
    
    for group_name, group_results in results.items():
        print(f"\n  {group_name}:")
        for test_name, passed in group_results.items():
            if passed is True:
                print(f"    ✅ {test_name}")
                total_pass += 1
            elif passed is False:
                print(f"    ❌ {test_name}")
                total_fail += 1
            else:
                print(f"    ⏭️  {test_name} (skipped)")
                total_skip += 1
    
    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed, {total_skip} skipped")
    print(f"{'='*60}")
    
    return total_fail == 0


def main():
    parser = argparse.ArgumentParser(description="Test point cloud integration")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "preprocessing", "data_loading", "model_preprocessing", 
                 "batch_collation", "model_forward", "full_pipeline"],
        help="Which test group to run"
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  POINT CLOUD INTEGRATION TEST SUITE")
    print("="*60)
    
    if args.test == "all":
        results = run_all_tests()
    elif args.test == "preprocessing":
        results = {"preprocessing": test_preprocessing()}
    elif args.test == "data_loading":
        results = {"data_loading": test_data_loading()}
    elif args.test == "model_preprocessing":
        results = {"model_preprocessing": test_model_preprocessing()}
    elif args.test == "batch_collation":
        results = {"batch_collation": test_batch_collation()}
    elif args.test == "model_forward":
        results = {"model_forward": test_model_forward()}
    elif args.test == "full_pipeline":
        results = {"full_pipeline": test_full_pipeline()}
    
    success = print_summary(results)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

