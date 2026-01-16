# Point Cloud Trajectory Integration - Implementation Complete

## Summary

This document describes the point cloud integration for robot training. The implementation enables accumulating point clouds from trajectory history and using them as an additional modality alongside images.

## What Was Implemented

### 1. VGGT Preprocessing Script
**File:** `scripts/precompute_vggt_depth.py`

- Processes VIDA video data to extract point clouds
- Uses VGGT (or dummy estimation for testing) to estimate depth
- Accumulates point clouds from ALL frames up to each keyframe
- Transforms points to robot-centered coordinates
- Saves to HDF5 files organized by house/episode/frame

### 2. Robot Dataset Updates
**File:** `olmo/data/robot_datasets.py`

- Added `use_point_cloud` and `point_cloud_dir` parameters to `RobotDataset`
- Added `_load_point_cloud()` method to load preprocessed point clouds
- Updated `_process_file_chunk()` to include point cloud data in examples

### 3. Training Script Updates
**File:** `launch_scripts/train_multitask_model.py`

- Added `--use_point_cloud` flag to enable point cloud processing
- Added `--point_cloud_dir` argument to specify preprocessed data location
- Configures `PointCloudBackboneConfig` when point clouds are enabled

### 4. Dataset Factory Updates
**File:** `olmo/data/get_dataset.py`

- Reads `ROBOT_USE_POINT_CLOUD` and `ROBOT_POINT_CLOUD_DIR` environment variables
- Passes point cloud settings to RobotDataset instances

### 5. Preprocessing Helper Script
**File:** `scripts/run_vggt_preprocessing.sh`

Convenience script for running VGGT preprocessing.

## How to Use

### Step 1: Run Preprocessing (One-time)

```bash
# Preprocess ObjectNav training data
./scripts/run_vggt_preprocessing.sh ObjectNavType train

# Preprocess other task types
./scripts/run_vggt_preprocessing.sh HardObjectNavType train
./scripts/run_vggt_preprocessing.sh SimpleExploreHouse train
```

Or run directly:
```bash
python scripts/precompute_vggt_depth.py \
    --input_dir /weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor/ObjectNavType/train \
    --output_dir /path/to/output \
    --max_points 50000 \
    --max_keyframes 10
```

### Step 2: Train with Point Clouds

```bash
torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py \
    robot_mixture \
    /path/to/checkpoint \
    --robot_memory_setting=SceneMemory \
    --use_point_cloud \
    --point_cloud_dir=/path/to/preprocessed/point_clouds \
    --save_folder=./robot_training_with_pointclouds \
    --global_batch_size=256 \
    --device_train_batch_size=4 \
    --duration=10000
```

## Architecture

```
Point Cloud Pipeline:
  1. Offline Preprocessing (VGGT):
     Video Frames → VGGT → Depth Maps → Point Clouds → Accumulate → HDF5

  2. Training Runtime:
     Load Point Cloud → Point Transformer V3 → Voxelize → Max Pool → Project → PC Tokens
     Load Image → Vision Backbone → Image Tokens
     
     [Text Tokens] [Image Tokens] [PC Tokens] → LLM → Action Prediction
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Preprocessing | Offline | Training speed, avoid GPU memory for VGGT |
| Accumulation | Full trajectory history | Better spatial understanding |
| Coordinate frame | Robot-centered | Egocentric navigation |
| Keyframes | Selected by `select_good_frames()` | Quality frames with valid data |

## Files Changed

| File | Changes |
|------|---------|
| `scripts/precompute_vggt_depth.py` | Complete rewrite for VIDA structure |
| `scripts/run_vggt_preprocessing.sh` | New convenience script |
| `olmo/data/robot_datasets.py` | Added point cloud loading |
| `olmo/data/get_dataset.py` | Added env var support |
| `launch_scripts/train_multitask_model.py` | Added CLI flags |

## Existing Infrastructure (Already Implemented)

These files were already implemented and didn't need changes:
- `olmo/nn/point_cloud_backbone.py` - Point Transformer + voxelization
- `olmo/models/molmo/molmo.py` - Model with point cloud backbone
- `olmo/models/molmo/model_preprocessor.py` - Token generation
- `olmo/models/molmo/collator.py` - Batch collation
- `olmo/tokenizer.py` - Point cloud special tokens

## Notes

- VGGT requires the `vggt` package. If not installed, dummy depth estimation is used.
- Point cloud preprocessing is I/O and compute intensive. Use GPU for faster processing.
- Point clouds are optional - training works without them if `--use_point_cloud` is not set.

