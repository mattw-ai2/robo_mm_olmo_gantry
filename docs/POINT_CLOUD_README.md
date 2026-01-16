# Point Cloud Integration for Robot Training

## Overview

This document describes the point cloud integration added to the robot training pipeline. The system accumulates 3D point clouds from trajectory history and uses them as an additional input modality alongside the 2x2 camera grid images.

## Background

### The Problem
The original robot training pipeline only uses a **single frame's 4-camera composite image**. This limits spatial understanding because:
- No memory of previously seen areas
- Cannot build a 3D map of the environment
- Each frame is processed independently

### The Solution
Add point cloud accumulation from **all frames in the trajectory up to the current frame**, transformed to **robot-centered coordinates**. This gives the model:
- 3D spatial memory of the environment
- Understanding of previously explored areas
- Egocentric representation that moves with the robot

## Trajectory Statistics

Based on the VIDA dataset (`31Jul2025_timebudget_05hz_FPIN_new_procthor`):
- **Trajectory lengths**: 19-77 frames
- **Mean length**: ~40 frames
- **Sampling rate**: 0.5 Hz (every 2 seconds)
- **Keyframes selected**: Up to 10 per trajectory (via `select_good_frames()`)

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │         OFFLINE PREPROCESSING           │
                          │            (Run Once)                   │
                          └─────────────────────────────────────────┘
                                          │
     ┌────────────────┐    ┌──────────────┴──────────────┐
     │  Video Frames  │───►│         VGGT Model          │
     │  (4 cameras)   │    │   (Depth + Extrinsics)      │
     └────────────────┘    └──────────────┬──────────────┘
                                          │
                           ┌──────────────▼──────────────┐
                           │  Depth → Point Cloud        │
                           │  (Per camera, per frame)    │
                           └──────────────┬──────────────┘
                                          │
                           ┌──────────────▼──────────────┐
                           │  Accumulate Trajectory      │
                           │  (Frame 0 to current)       │
                           └──────────────┬──────────────┘
                                          │
                           ┌──────────────▼──────────────┐
                           │  Transform to Robot Frame   │
                           │  (Egocentric coordinates)   │
                           └──────────────┬──────────────┘
                                          │
                           ┌──────────────▼──────────────┐
                           │     Save to HDF5            │
                           │ (Per house/episode/frame)   │
                           └──────────────────────────────┘

                          ┌─────────────────────────────────────────┐
                          │           TRAINING RUNTIME              │
                          └─────────────────────────────────────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         │                                │                                │
         ▼                                ▼                                ▼
┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
│  Load Image     │             │Load Point Cloud │             │  Load Text      │
│  (2x2 grid)     │             │  (Accumulated)  │             │  (Prompt)       │
└────────┬────────┘             └────────┬────────┘             └────────┬────────┘
         │                                │                                │
         ▼                                ▼                                │
┌─────────────────┐             ┌─────────────────┐                       │
│ Vision Backbone │             │Point Transformer│                       │
│   (CLIP/SigLIP) │             │      V3         │                       │
└────────┬────────┘             └────────┬────────┘                       │
         │                                │                                │
         ▼                                ▼                                │
┌─────────────────┐             ┌─────────────────┐                       │
│  Image Tokens   │             │  Voxelize +     │                       │
│                 │             │  Max Pool       │                       │
└────────┬────────┘             └────────┬────────┘                       │
         │                                │                                │
         │                                ▼                                │
         │                      ┌─────────────────┐                       │
         │                      │ Linear Project  │                       │
         │                      │ → PC Tokens     │                       │
         │                      └────────┬────────┘                       │
         │                                │                                │
         └────────────────┬───────────────┴────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────────────────────────────────┐
              │  [Text Tokens] [Image Tokens] [PC Tokens] [Text]  │
              │                                                   │
              │                    LLM Backbone                   │
              │                                                   │
              └───────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────────────────────────────────┐
              │              Action Prediction                    │
              │  (Point coordinates or "DONE")                   │
              └───────────────────────────────────────────────────┘
```

## Files Changed

### New Files
| File | Description |
|------|-------------|
| `scripts/precompute_vggt_depth.py` | VGGT preprocessing script for VIDA dataset |
| `scripts/run_vggt_preprocessing.sh` | Convenience wrapper script |
| `docs/POINT_CLOUD_README.md` | This documentation |
| `plans/POINT_CLOUD_IMPLEMENTATION.md` | Implementation summary |

### Modified Files
| File | Changes |
|------|---------|
| `olmo/data/robot_datasets.py` | Added `use_point_cloud`, `point_cloud_dir` params; Added `_load_point_cloud()` method |
| `olmo/data/get_dataset.py` | Added env var support for `ROBOT_USE_POINT_CLOUD`, `ROBOT_POINT_CLOUD_DIR` |
| `launch_scripts/train_multitask_model.py` | Added `--use_point_cloud`, `--point_cloud_dir` CLI flags; Added model config for point cloud backbone |

### Pre-existing Files (No changes needed)
| File | Purpose |
|------|---------|
| `olmo/nn/point_cloud_backbone.py` | Point Transformer V3 + voxelization + projection |
| `olmo/models/molmo/molmo.py` | Model with optional point cloud backbone |
| `olmo/models/molmo/model_preprocessor.py` | Token generation including PC tokens |
| `olmo/models/molmo/collator.py` | Batch collation with PC padding |
| `olmo/tokenizer.py` | Special tokens: `<pc_start>`, `<pc_patch>`, `<pc_end>` |

## Usage

### Step 1: Preprocess Point Clouds (One-time)

```bash
# Using the convenience script
./scripts/run_vggt_preprocessing.sh ObjectNavType train
./scripts/run_vggt_preprocessing.sh HardObjectNavType train
./scripts/run_vggt_preprocessing.sh SimpleExploreHouse train

# Or directly
python scripts/precompute_vggt_depth.py \
    --input_dir /weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor/ObjectNavType/train \
    --output_dir /weka/prior/mattw/data/robot_point_clouds/ObjectNavType/train \
    --max_points 50000 \
    --max_keyframes 10 \
    --device cuda
```

**Output structure:**
```
output_dir/
├── 000000/
│   └── point_clouds.hdf5
│       ├── 0/           # Episode 0
│       │   ├── 5/       # Frame 5
│       │   │   ├── point_cloud      # [N, 3] xyz coordinates
│       │   │   └── point_cloud_mask # [N] validity mask
│       │   ├── 12/      # Frame 12
│       │   └── ...
│       ├── 1/           # Episode 1
│       └── ...
├── 000001/
└── ...
```

### Step 2: Train with Point Clouds

```bash
torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py \
    robot_mixture \
    /path/to/molmo-checkpoint \
    --robot_memory_setting=SceneMemory \
    --use_point_cloud \
    --point_cloud_dir=/weka/prior/mattw/data/robot_point_clouds \
    --save_folder=./robot_training_with_pointclouds \
    --global_batch_size=256 \
    --device_train_batch_size=4 \
    --duration=10000 \
    --wandb.name=robot-pc-training
```

### Step 3: Inference with Point Clouds

The model will automatically use point clouds if:
1. The model was trained with `--use_point_cloud`
2. Point cloud data is available in the input

## Testing and Debugging

### Quick Import Test
```python
# Test that all components import correctly
from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig
from olmo.data.robot_datasets import RobotDataset
from olmo.models.molmo.molmo import MolmoConfig, Molmo

# Verify RobotDataset has new parameters
import inspect
sig = inspect.signature(RobotDataset.__init__)
assert 'use_point_cloud' in sig.parameters
assert 'point_cloud_dir' in sig.parameters
print("✓ All imports successful")
```

### Test Point Cloud Loading
```python
import numpy as np
import h5py

# Check preprocessed point cloud file
pc_path = "/path/to/point_clouds/000000/point_clouds.hdf5"
with h5py.File(pc_path, 'r') as f:
    print("Episodes:", list(f.keys()))
    episode = f['0']
    print("Frames:", list(episode.keys()))
    frame = episode['5']
    pc = np.array(frame['point_cloud'])
    mask = np.array(frame['point_cloud_mask'])
    print(f"Point cloud shape: {pc.shape}")  # Should be [N, 3]
    print(f"Valid points: {mask.sum()} / {len(mask)}")
```

### Test RobotDataset with Point Clouds
```python
import os
os.environ["ROBOT_USE_POINT_CLOUD"] = "1"
os.environ["ROBOT_POINT_CLOUD_DIR"] = "/path/to/point_clouds"

from olmo.data.robot_datasets import RobotDataset

dataset = RobotDataset(
    task_type="ObjectNav",
    split="train",
    memory_setting="SceneMemory",
    use_point_cloud=True,
    point_cloud_dir="/path/to/point_clouds"
)

# Check if point clouds are loaded
example = dataset[0]
if "point_cloud" in example:
    print(f"✓ Point cloud loaded: shape={example['point_cloud'].shape}")
else:
    print("✗ No point cloud in example")
```

### Debug Preprocessing
```bash
# Process only a few houses for testing
python scripts/precompute_vggt_depth.py \
    --input_dir /path/to/input \
    --output_dir /path/to/output \
    --num_houses 2 \
    --max_keyframes 3 \
    --device cpu  # Use CPU to debug without GPU
```

### Verify Point Cloud Backbone
```python
import torch
from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig

config = PointCloudBackboneConfig(
    voxel_size=0.1,
    grid_range=10.0,
    ptv3_channels=512,
)

backbone = PointCloudBackbone(config, llm_dim=4096)

# Test forward pass
points = torch.randn(2, 1000, 3)  # [batch, num_points, xyz]
mask = torch.ones(2, 1000, dtype=torch.bool)

with torch.no_grad():
    tokens, valid_mask = backbone(points, mask)
    
print(f"Output tokens shape: {tokens.shape}")  # [B, num_voxels, llm_dim]
print(f"Valid voxels: {valid_mask.sum(dim=1).tolist()}")
```

## Key Parameters

### Preprocessing Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_points` | 50000 | Max points in accumulated cloud per keyframe |
| `--max_keyframes` | 10 | Max keyframes to process per episode |
| `--device` | cuda | Device for VGGT inference |
| `--num_houses` | None | Limit houses for testing (None = all) |

### Model Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `voxel_size` | 0.1 | Voxel size in meters |
| `grid_range` | 10.0 | Grid extent [-R, R] in meters |
| `ptv3_channels` | 512 | Point Transformer feature dimension |
| `ptv3_num_layers` | 4 | Number of transformer layers |
| `ptv3_num_heads` | 8 | Number of attention heads |

### Training Parameters
| Parameter | Description |
|-----------|-------------|
| `--use_point_cloud` | Enable point cloud processing |
| `--point_cloud_dir` | Path to preprocessed point clouds |

## Common Issues

### 1. "Point cloud not found for frame X"
- The preprocessing only creates point clouds for selected keyframes
- The loader finds the closest available frame if exact frame not found

### 2. "VGGT not installed"
- Install with: `pip install vggt`
- Or preprocessing will use dummy depth (random values for testing)

### 3. Out of Memory during preprocessing
- Reduce `--max_points`
- Process fewer houses at a time with `--num_houses`

### 4. Slow training with point clouds
- Point cloud backbone adds ~10-15% overhead
- Reduce `ptv3_num_layers` for faster training
- Reduce `max_point_cloud_tokens` in preprocessor

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Preprocessing | Offline | Avoid GPU memory for VGGT during training |
| Accumulation | Full trajectory | Better spatial memory than sliding window |
| Coordinate frame | Robot-centered | Egocentric for navigation |
| Keyframe selection | `select_good_frames()` | Quality frames with valid points |
| Temporal subsampling | Every 2nd frame | Balance coverage vs compute |

## Future Improvements

1. **Online VGGT** - Run VGGT during training for more flexibility
2. **Sliding window** - Limit accumulation to last N frames
3. **Voxel features** - Add RGB color to point clouds
4. **Multi-scale voxelization** - Different voxel sizes for near/far
5. **Attention-based fusion** - Cross-attention between image and PC tokens

---

## Related Documentation

| Document | Description | Date |
|----------|-------------|------|
| [Testing Commands](./2026-01-13_point_cloud_testing_commands.md) | Modular test commands for each pipeline component | 2026-01-13 |
| [Testing Plan](./POINT_CLOUD_TESTING_PLAN.md) | Detailed testing plan with code snippets | 2026-01-13 |
| [Implementation Summary](../plans/POINT_CLOUD_IMPLEMENTATION.md) | Summary of what was implemented | 2026-01-13 |

