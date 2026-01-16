# Point Cloud Pipeline Testing Commands
**Date:** 2026-01-13  
**Theme:** Modular testing commands for each component of the point cloud training pipeline

---

## Quick Reference

```bash
cd /weka/prior/mattw/robo_mm_olmo && source .venv/bin/activate

# Run automated test suite
python tests/test_point_cloud_integration.py --test all

# Run specific test groups
python tests/test_point_cloud_integration.py --test preprocessing
python tests/test_point_cloud_integration.py --test data_loading
python tests/test_point_cloud_integration.py --test model_forward
python tests/test_point_cloud_integration.py --test full_pipeline
```

---

## Pipeline Components

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. Backbone  │ → │ 2. Tokenize  │ → │ 3. Dataset   │ → │ 4. Collator  │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 8. Gradients │ ← │ 7. Training  │ ← │ 6. Molmo     │ ← │ 5. Preproc   │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

---

## 1. Test Point Cloud Backbone

**Purpose:** Verify PointCloudBackbone can process points → tokens

```bash
python -c "
import torch
from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig

config = PointCloudBackboneConfig(voxel_size=0.2, grid_range=5.0, ptv3_channels=128, ptv3_num_layers=1)
backbone = PointCloudBackbone(config, llm_dim=1024)

points = torch.randn(2, 500, 3)
mask = torch.ones(2, 500, dtype=torch.bool)

tokens, valid_mask = backbone(points, mask)
print(f'✅ Backbone: points {points.shape} → tokens {tokens.shape}')
print(f'   Valid voxels per sample: {valid_mask.sum(dim=1).tolist()}')
"
```

**Expected output:**
```
✅ Backbone: points torch.Size([2, 500, 3]) → tokens torch.Size([2, N, 1024])
   Valid voxels per sample: [X, Y]
```

---

## 2. Test Point Cloud Tokens

**Purpose:** Verify special tokens exist in vocabulary

```bash
python -c "
from olmo.tokenizer import Tokenizer

tokenizer = Tokenizer.from_pretrained('allenai/Molmo-7B-D-0924')
vocab = tokenizer.vocab

pc_tokens = ['<|pc_start|>', '<|pc_patch|>', '<|pc_end|>']
for tok in pc_tokens:
    if tok in vocab:
        print(f'✅ {tok} → ID {vocab[tok]}')
    else:
        print(f'❌ {tok} not in vocabulary')
"
```

**Expected output:**
```
✅ <|pc_start|> → ID XXXXX
✅ <|pc_patch|> → ID XXXXX
✅ <|pc_end|> → ID XXXXX
```

---

## 3. Test Point Cloud Tokenization in Preprocessor

**Purpose:** Verify MolmoPreprocessor handles point cloud tokenization

```bash
python -c "
from olmo.models.molmo.model_preprocessor import MolmoPreprocessor
import inspect

source = inspect.getsource(MolmoPreprocessor)

checks = [
    ('point_cloud', 'point_cloud handling'),
    ('pc_start', 'pc_start token'),
    ('pc_patch', 'pc_patch token'),
    ('pc_end', 'pc_end token'),
]

for pattern, desc in checks:
    if pattern in source.lower():
        print(f'✅ MolmoPreprocessor has {desc}')
    else:
        print(f'⚠️  MolmoPreprocessor missing {desc}')
"
```

**Expected output:**
```
✅ MolmoPreprocessor has point_cloud handling
✅ MolmoPreprocessor has pc_start token
✅ MolmoPreprocessor has pc_patch token
✅ MolmoPreprocessor has pc_end token
```

---

## 4. Test RobotDataset Point Cloud Loading

**Purpose:** Verify RobotDataset has point cloud parameters and loading method

```bash
python -c "
from olmo.data.robot_datasets import RobotDataset
import inspect

sig = inspect.signature(RobotDataset.__init__)
params = list(sig.parameters.keys())

print('RobotDataset parameters:')
for p in ['use_point_cloud', 'point_cloud_dir']:
    if p in params:
        print(f'  ✅ {p}')
    else:
        print(f'  ❌ {p} missing')

if hasattr(RobotDataset, '_load_point_cloud'):
    print('  ✅ _load_point_cloud method')
else:
    print('  ❌ _load_point_cloud method missing')
"
```

**Expected output:**
```
RobotDataset parameters:
  ✅ use_point_cloud
  ✅ point_cloud_dir
  ✅ _load_point_cloud method
```

---

## 5. Test Collator Point Cloud Batching

**Purpose:** Verify MMCollator handles point cloud padding/batching

```bash
python -c "
from olmo.models.molmo.collator import MMCollator
import inspect

source = inspect.getsource(MMCollator)

if 'point_cloud' in source.lower():
    print('✅ MMCollator handles point clouds')
else:
    print('⚠️  MMCollator may not handle point clouds')

if 'pad' in source.lower() and 'point' in source.lower():
    print('✅ MMCollator has point cloud padding')
"
```

**Expected output:**
```
✅ MMCollator handles point clouds
✅ MMCollator has point cloud padding
```

---

## 6. Test Molmo Model Integration

**Purpose:** Verify MolmoConfig accepts point_cloud_backbone

```bash
python -c "
from olmo.models.molmo.molmo import MolmoConfig
import inspect

source = inspect.getsource(MolmoConfig)

if 'point_cloud_backbone' in source:
    print('✅ MolmoConfig has point_cloud_backbone field')
else:
    print('❌ MolmoConfig missing point_cloud_backbone')
"
```

**Expected output:**
```
✅ MolmoConfig has point_cloud_backbone field
```

---

## 7. Test Training Script Args

**Purpose:** Verify training script has point cloud CLI arguments

```bash
python -c "
from pathlib import Path
content = Path('launch_scripts/train_multitask_model.py').read_text()

checks = [
    ('--use_point_cloud', 'use_point_cloud flag'),
    ('--point_cloud_dir', 'point_cloud_dir arg'),
    ('PointCloudBackboneConfig', 'backbone config'),
    ('ROBOT_USE_POINT_CLOUD', 'env var for use_point_cloud'),
    ('ROBOT_POINT_CLOUD_DIR', 'env var for point_cloud_dir'),
]

for pattern, desc in checks:
    if pattern in content:
        print(f'✅ {desc}')
    else:
        print(f'⚠️  missing {desc}')
"
```

**Expected output:**
```
✅ use_point_cloud flag
✅ point_cloud_dir arg
✅ backbone config
✅ env var for use_point_cloud
✅ env var for point_cloud_dir
```

---

## 8. Test Gradient Flow

**Purpose:** Verify gradients flow through point cloud backbone

```bash
python -c "
import torch
from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig

config = PointCloudBackboneConfig(voxel_size=0.2, grid_range=5.0, ptv3_channels=128, ptv3_num_layers=1)
backbone = PointCloudBackbone(config, llm_dim=1024)
backbone.train()

points = torch.randn(2, 500, 3)
mask = torch.ones(2, 500, dtype=torch.bool)

tokens, valid_mask = backbone(points, mask)
loss = tokens.mean()
loss.backward()

has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())
if has_grads:
    print('✅ Gradients flow through backbone')
else:
    print('❌ No gradients in backbone')
"
```

**Expected output:**
```
✅ Gradients flow through backbone
```

---

## 9. Full Model Forward Pass (Encoder → Molmo LLM → Output)

**Purpose:** Test complete forward pass through the entire model with point clouds

```bash
# Test with trainable point cloud backbone
python tests/test_full_molmo_with_pointcloud.py \
    --checkpoint /path/to/molmo-checkpoint \
    --voxel_size 0.2 \
    --grid_range 5.0 \
    --ptv3_channels 256

# Test with FROZEN point cloud backbone (no gradients)
python tests/test_full_molmo_with_pointcloud.py \
    --checkpoint /path/to/molmo-checkpoint \
    --freeze_pc_backbone
```

**What it tests:**
```
Point Cloud [B, N, 3]
     ↓
PointCloudBackbone (voxelize → PTv3 → project)
     ↓
PC Tokens [B, num_voxels, d_model]
     ↓
Molmo LLM (with image + text tokens)
     ↓
Output Logits [B, seq_len, vocab_size]
     ↓
Gradient flow back through entire model
```

---

## 10. Full Training Pipeline Dry Run

**Purpose:** Test training with point clouds enabled (short duration)

```bash
# Train with trainable point cloud backbone
torchrun --nproc-per-node=1 launch_scripts/train_multitask_model.py \
    robot_mixture \
    /path/to/molmo-checkpoint \
    --use_point_cloud \
    --point_cloud_dir=/path/to/point_clouds \
    --duration=5 \
    --global_batch_size=2 \
    --device_train_batch_size=1 \
    --dry_run

# Train with FROZEN point cloud backbone (only fine-tune LLM)
torchrun --nproc-per-node=1 launch_scripts/train_multitask_model.py \
    robot_mixture \
    /path/to/molmo-checkpoint \
    --use_point_cloud \
    --point_cloud_dir=/path/to/point_clouds \
    --freeze_point_cloud_backbone \
    --duration=5 \
    --global_batch_size=2 \
    --device_train_batch_size=1
```

---

## Run All Tests Script

Save and run this as a single script:

```bash
#!/bin/bash
# test_point_cloud_pipeline.sh

cd /weka/prior/mattw/robo_mm_olmo
source .venv/bin/activate

echo "=========================================="
echo "  POINT CLOUD PIPELINE TESTS"
echo "=========================================="

echo -e "\n1. Testing PointCloudBackbone..."
python -c "
import torch
from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig
config = PointCloudBackboneConfig(voxel_size=0.2, grid_range=5.0, ptv3_channels=128, ptv3_num_layers=1)
backbone = PointCloudBackbone(config, llm_dim=1024)
points = torch.randn(2, 500, 3)
mask = torch.ones(2, 500, dtype=torch.bool)
tokens, valid_mask = backbone(points, mask)
print(f'✅ Backbone: {points.shape} → {tokens.shape}')
" 2>/dev/null || echo "❌ Backbone test failed"

echo -e "\n2. Testing RobotDataset..."
python -c "
from olmo.data.robot_datasets import RobotDataset
import inspect
sig = inspect.signature(RobotDataset.__init__)
params = list(sig.parameters.keys())
assert 'use_point_cloud' in params and 'point_cloud_dir' in params
print('✅ RobotDataset has point cloud params')
" 2>/dev/null || echo "❌ RobotDataset test failed"

echo -e "\n3. Testing MolmoConfig..."
python -c "
from olmo.models.molmo.molmo import MolmoConfig
import inspect
assert 'point_cloud' in inspect.getsource(MolmoConfig)
print('✅ MolmoConfig has point_cloud_backbone')
" 2>/dev/null || echo "❌ MolmoConfig test failed"

echo -e "\n4. Testing training script..."
python -c "
content = open('launch_scripts/train_multitask_model.py').read()
assert '--use_point_cloud' in content and '--point_cloud_dir' in content
print('✅ Training script has point cloud args')
" 2>/dev/null || echo "❌ Training script test failed"

echo -e "\n5. Testing gradient flow..."
python -c "
import torch
from olmo.nn.point_cloud_backbone import PointCloudBackbone, PointCloudBackboneConfig
config = PointCloudBackboneConfig(voxel_size=0.2, grid_range=5.0, ptv3_channels=128, ptv3_num_layers=1)
backbone = PointCloudBackbone(config, llm_dim=1024)
backbone.train()
tokens, _ = backbone(torch.randn(2, 500, 3), torch.ones(2, 500, dtype=torch.bool))
tokens.mean().backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())
print('✅ Gradients flow correctly')
" 2>/dev/null || echo "❌ Gradient test failed"

echo -e "\n=========================================="
echo "  TESTS COMPLETE"
echo "=========================================="
```

---

## Debugging Tips

| Symptom | Check |
|---------|-------|
| `ModuleNotFoundError` | Activate venv: `source .venv/bin/activate` |
| Token not found | Check `olmo/tokenizer.py` for PC token definitions |
| Shape mismatch | Verify `max_points` and `voxel_size` settings |
| No gradients | Ensure `backbone.train()` called |
| OOM | Reduce `ptv3_channels` or `ptv3_num_layers` |

---

## Related Documentation

- [Point Cloud README](./POINT_CLOUD_README.md) - Full implementation guide
- [Testing Plan](./POINT_CLOUD_TESTING_PLAN.md) - Detailed testing plan
- [Implementation Summary](../plans/POINT_CLOUD_IMPLEMENTATION.md) - What was implemented

