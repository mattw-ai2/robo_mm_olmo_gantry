# Point Cloud Integration Testing Plan

## Quick Reference

**Run all tests:**
```bash
cd /weka/prior/mattw/robo_mm_olmo && source .venv/bin/activate
python tests/test_point_cloud_integration.py --test all
```

**Run individual test groups:**
```bash
python tests/test_point_cloud_integration.py --test preprocessing
python tests/test_point_cloud_integration.py --test data_loading
python tests/test_point_cloud_integration.py --test model_forward
python tests/test_point_cloud_integration.py --test full_pipeline
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE PREPROCESSING                              │
│  Video → VGGT → Depth → Unproject → Accumulate → Robot Frame → Save HDF5   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA LOADING                                     │
│  RobotDataset → _load_point_cloud() → example["point_cloud"]                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODEL PREPROCESSING                                 │
│  MolmoPreprocessor → Tokenize → [pc_start][pc_patch...][pc_end]             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BATCH COLLATION                                   │
│  MMCollator → Pad point clouds → Batch tensors                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL FORWARD PASS                                 │
│  PointCloudBackbone → Voxelize → PTv3 → Project → PC Tokens → Molmo LLM    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TRAINING LOOP                                    │
│  Loss computation → Backward → Gradients flow through PC backbone           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. PREPROCESSING TESTS

### 1.1 Depth → Point Cloud Unprojection
**File:** `scripts/precompute_vggt_depth.py`  
**Function:** `unproject_depth_to_points(depth, intrinsics)`

**Test:**
```python
import numpy as np
from scripts.precompute_vggt_depth import unproject_depth_to_points

# Create synthetic depth
depth = np.random.uniform(0.5, 5.0, (64, 64)).astype(np.float32)
intrinsics = np.array([[500, 0, 32], [0, 500, 32], [0, 0, 1]], dtype=np.float32)

points = unproject_depth_to_points(depth, intrinsics)
assert points.shape == (64, 64, 3)
assert not np.any(np.isnan(points))
```

### 1.2 Trajectory Accumulation
**File:** `scripts/precompute_vggt_depth.py`  
**Function:** `accumulate_trajectory_points(all_points, all_extrinsics, max_points)`

**Test:**
```python
from scripts.precompute_vggt_depth import accumulate_trajectory_points

# Create 5 frames of synthetic points
all_points = [np.random.randn(1000, 3).astype(np.float32) for _ in range(5)]
all_extrinsics = [np.eye(4, dtype=np.float32) for _ in range(5)]

accumulated, mask = accumulate_trajectory_points(all_points, all_extrinsics, max_points=3000)
assert accumulated.shape[0] <= 3000
assert accumulated.shape[1] == 3
```

### 1.3 Robot-Centered Transform
**File:** `scripts/precompute_vggt_depth.py`  
**Function:** `transform_to_robot_frame(points_world, robot_pose)`

**Test:**
```python
from scripts.precompute_vggt_depth import transform_to_robot_frame

points_world = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
robot_pose = np.eye(4, dtype=np.float32)
robot_pose[:3, 3] = [1, 0, 0]  # Robot at [1, 0, 0]

points_robot = transform_to_robot_frame(points_world, robot_pose)
assert points_robot.shape == points_world.shape
# Point [1,0,0] should be at origin in robot frame
```

### 1.4 HDF5 Save/Load
**Test:**
```python
import h5py
import numpy as np
import tempfile

points = np.random.randn(1000, 3).astype(np.float32)
mask = np.ones(1000, dtype=bool)

with tempfile.NamedTemporaryFile(suffix='.hdf5') as f:
    # Save
    with h5py.File(f.name, 'w') as hf:
        ep = hf.create_group("0")
        frame = ep.create_group("5")
        frame.create_dataset("point_cloud", data=points, compression='gzip')
        frame.create_dataset("point_cloud_mask", data=mask, compression='gzip')
    
    # Load
    with h5py.File(f.name, 'r') as hf:
        loaded = np.array(hf["0"]["5"]["point_cloud"])
    
    assert np.allclose(points, loaded)
```

---

## 2. DATA LOADING TESTS

### 2.1 RobotDataset Configuration
**File:** `olmo/data/robot_datasets.py`

**Test:**
```python
from olmo.data.robot_datasets import RobotDataset
import inspect

sig = inspect.signature(RobotDataset.__init__)
params = list(sig.parameters.keys())

assert 'use_point_cloud' in params, "Missing use_point_cloud parameter"
assert 'point_cloud_dir' in params, "Missing point_cloud_dir parameter"
```

### 2.2 Point Cloud Loading Method
**File:** `olmo/data/robot_datasets.py`  
**Method:** `_load_point_cloud(house_id, episode_id, frame_idx)`

**Test:**
```python
from olmo.data.robot_datasets import RobotDataset

assert hasattr(RobotDataset, '_load_point_cloud')
```

### 2.3 Environment Variable Support
**File:** `olmo/data/get_dataset.py`

**Test:**
```python
import os
os.environ["ROBOT_USE_POINT_CLOUD"] = "1"
os.environ["ROBOT_POINT_CLOUD_DIR"] = "/path/to/point_clouds"

from olmo.data.get_dataset import get_dataset_by_name
# Should pass these to RobotDataset
```

---

## 3. MODEL PREPROCESSING TESTS

### 3.1 Point Cloud Tokens
**File:** `olmo/tokenizer.py`

**Test:**
```python
from olmo.tokenizer import (
    POINT_CLOUD_START_TOKEN,
    POINT_CLOUD_PATCH_TOKEN,
    POINT_CLOUD_END_TOKEN,
)

print(f"PC_START: {POINT_CLOUD_START_TOKEN}")
print(f"PC_PATCH: {POINT_CLOUD_PATCH_TOKEN}")
print(f"PC_END: {POINT_CLOUD_END_TOKEN}")
```

### 3.2 MolmoPreprocessor Point Cloud Handling
**File:** `olmo/models/molmo/model_preprocessor.py`

**Test:**
```python
from olmo.models.molmo.model_preprocessor import MolmoPreprocessor
import inspect

source = inspect.getsource(MolmoPreprocessor)
assert 'point_cloud' in source, "Preprocessor doesn't handle point clouds"
```

---

## 4. BATCH COLLATION TESTS

### 4.1 MMCollator Point Cloud Batching
**File:** `olmo/models/molmo/collator.py`

**Test:**
```python
from olmo.models.molmo.collator import MMCollator
import inspect

source = inspect.getsource(MMCollator)
assert 'point_cloud' in source.lower(), "Collator doesn't handle point clouds"
```

---

## 5. MODEL FORWARD PASS TESTS

### 5.1 PointCloudBackboneConfig
**File:** `olmo/nn/point_cloud_backbone.py`

**Test:**
```python
from olmo.nn.point_cloud_backbone import PointCloudBackboneConfig

config = PointCloudBackboneConfig(
    voxel_size=0.1,
    grid_range=10.0,
    ptv3_channels=256,
)
print(f"voxel_size={config.voxel_size}, channels={config.ptv3_channels}")
```

### 5.2 PointCloudBackbone Initialization
**File:** `olmo/nn/point_cloud_backbone.py`

**Test:**
```python
import torch
from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig

config = PointCloudBackboneConfig(
    voxel_size=0.2,
    grid_range=5.0,
    ptv3_channels=128,
    ptv3_num_layers=1,
)

backbone = PointCloudBackbone(config, llm_dim=1024)
num_params = sum(p.numel() for p in backbone.parameters())
print(f"Parameters: {num_params:,}")
```

### 5.3 PointCloudBackbone Forward Pass
**Test:**
```python
import torch
from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig

config = PointCloudBackboneConfig(voxel_size=0.2, grid_range=5.0, ptv3_channels=128)
backbone = PointCloudBackbone(config, llm_dim=1024)
backbone.eval()

points = torch.randn(2, 500, 3)  # [batch, num_points, xyz]
mask = torch.ones(2, 500, dtype=torch.bool)

with torch.no_grad():
    tokens, valid_mask = backbone(points, mask)

print(f"Input: {points.shape} → Output: {tokens.shape}")
print(f"Valid voxels: {valid_mask.sum(dim=1).tolist()}")
```

### 5.4 MolmoConfig Point Cloud Backbone
**File:** `olmo/models/molmo/molmo.py`

**Test:**
```python
from olmo.models.molmo.molmo import MolmoConfig
import inspect

source = inspect.getsource(MolmoConfig)
assert 'point_cloud_backbone' in source, "MolmoConfig missing point_cloud_backbone"
```

---

## 6. TRAINING LOOP TESTS

### 6.1 Training Script Arguments
**File:** `launch_scripts/train_multitask_model.py`

**Test:**
```python
from pathlib import Path

content = Path("launch_scripts/train_multitask_model.py").read_text()
assert "--use_point_cloud" in content
assert "--point_cloud_dir" in content
assert "PointCloudBackboneConfig" in content
```

### 6.2 Gradient Flow
**Test:**
```python
import torch
from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig

config = PointCloudBackboneConfig(voxel_size=0.2, grid_range=5.0, ptv3_channels=128)
backbone = PointCloudBackbone(config, llm_dim=1024)
backbone.train()

points = torch.randn(2, 500, 3)
mask = torch.ones(2, 500, dtype=torch.bool)

tokens, valid_mask = backbone(points, mask)
loss = tokens.mean()
loss.backward()

# Check gradients exist
has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 
                for p in backbone.parameters())
assert has_grads, "No gradients flowing through backbone"
```

---

## Manual Integration Test

**Full end-to-end test with real data:**

```bash
# 1. Preprocess a small subset
python scripts/precompute_vggt_depth.py \
    --input_dir /weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor/ObjectNavType/train \
    --output_dir /tmp/test_point_clouds \
    --num_houses 1 \
    --max_keyframes 3

# 2. Test loading
python -c "
import os
os.environ['ROBOT_USE_POINT_CLOUD'] = '1'
os.environ['ROBOT_POINT_CLOUD_DIR'] = '/tmp/test_point_clouds'

from olmo.data.robot_datasets import RobotDataset
ds = RobotDataset(task_type='ObjectNav', split='train', use_point_cloud=True, point_cloud_dir='/tmp/test_point_clouds')
example = ds[0]
print('Keys:', list(example.keys()))
if 'point_cloud' in example:
    print('Point cloud shape:', example['point_cloud'].shape)
"

# 3. Test training (dry run)
torchrun --nproc-per-node=1 launch_scripts/train_multitask_model.py \
    robot_mixture /path/to/checkpoint \
    --use_point_cloud \
    --point_cloud_dir=/tmp/test_point_clouds \
    --duration=10 \
    --dry_run
```

---

## Debugging Checklist

| Issue | Check |
|-------|-------|
| Point cloud not loaded | Verify HDF5 file exists at expected path |
| Wrong shape | Check `max_points` in preprocessing |
| NaN in points | Check depth estimation / intrinsics |
| OOM in backbone | Reduce `ptv3_channels` or `ptv3_num_layers` |
| No gradients | Ensure `backbone.train()` is called |
| Slow training | Point cloud adds ~10-15% overhead |

