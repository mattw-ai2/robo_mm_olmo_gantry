# Environment Variables Reference

Complete reference for environment variables used in Molmo.

## Data Storage

### MOLMO_DATA_DIR

**Purpose:** Primary data directory for datasets and cached files.

**Default:** None (must be set)

**Usage:**
```bash
export MOLMO_DATA_DIR=/path/to/data
```

**What it stores:**
- `torch_datasets/`: Academic benchmarks
- `video_datasets/`: Video datasets
- `robot_datasets/`: Robot navigation data
- Preprocessed dataset files

### HF_HOME

**Purpose:** HuggingFace cache directory for models and datasets.

**Default:** `~/.cache/huggingface`

**Usage:**
```bash
export HF_HOME=/path/to/huggingface/cache
```

**What it stores:**
- Downloaded HuggingFace datasets
- Model checkpoints
- Tokenizers

### HF_DATASETS_CACHE

**Purpose:** Specific cache for HuggingFace datasets (subset of HF_HOME).

**Default:** `$HF_HOME/datasets`

**Usage:**
```bash
export HF_DATASETS_CACHE=/path/to/datasets/cache
```

## HuggingFace Configuration

### HF_ACCESS_TOKEN

**Purpose:** Authentication token for HuggingFace Hub.

**Usage:**
```bash
export HF_ACCESS_TOKEN=hf_your_token_here
```

**When needed:**
- Accessing gated models
- Uploading to HuggingFace Hub
- Private repositories

### HF_DATASETS_OFFLINE

**Purpose:** Use only cached datasets, no network requests.

**Values:** `1` (offline) or `0` (online)

**Default:** `0`

**Usage:**
```bash
export HF_DATASETS_OFFLINE=1  # Offline mode
```

**Benefits:**
- Faster data loading
- No rate limiting
- Works without internet
- Reduces unnecessary network requests

## Training Configuration

### WANDB_API_KEY

**Purpose:** Weights & Biases API key for logging.

**Usage:**
```bash
export WANDB_API_KEY=your_key_here
```

### WANDB_ENTITY

**Purpose:** W&B entity (username or team name).

**Usage:**
```bash
export WANDB_ENTITY=my-team
```

### WANDB_PROJECT

**Purpose:** W&B project name.

**Usage:**
```bash
export WANDB_PROJECT=molmo-training
```

### WANDB_MODE

**Purpose:** W&B operation mode.

**Values:**
- `online`: Full logging (default)
- `offline`: Log locally, sync later
- `disabled`: No logging

**Usage:**
```bash
export WANDB_MODE=offline
```

## Distributed Training

### MASTER_ADDR

**Purpose:** Address of master node in distributed training.

**Usage:**
```bash
export MASTER_ADDR=192.168.1.100
```

**When needed:** Multi-node training

### MASTER_PORT

**Purpose:** Port for master node communication.

**Default:** Set by torchrun

**Usage:**
```bash
export MASTER_PORT=29500
```

### NCCL_TIMEOUT_MINUTES

**Purpose:** Timeout for NCCL operations.

**Default:** `10`

**Usage:**
```bash
export NCCL_TIMEOUT_MINUTES=30  # Increase for slow networks
```

### NCCL_DEBUG

**Purpose:** NCCL debugging output level.

**Values:** `VERSION`, `WARN`, `INFO`, `TRACE`

**Usage:**
```bash
export NCCL_DEBUG=INFO
```

### NCCL_DEBUG_SUBSYS

**Purpose:** Which NCCL subsystems to debug.

**Values:** `ALL`, `INIT`, `COLL`, `P2P`, `NET`

**Usage:**
```bash
export NCCL_DEBUG_SUBSYS=ALL
```

### NCCL_SOCKET_IFNAME

**Purpose:** Network interface for NCCL communication.

**Usage:**
```bash
export NCCL_SOCKET_IFNAME=eth0
```

## PyTorch Configuration

### CUDA_VISIBLE_DEVICES

**Purpose:** Control which GPUs are visible to PyTorch.

**Usage:**
```bash
# Use only GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# Use all GPUs (default)
export CUDA_VISIBLE_DEVICES=""
```

### PYTORCH_CUDA_ALLOC_CONF

**Purpose:** Configure CUDA memory allocator.

**Usage:**
```bash
# Expandable memory segments (helps with fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Garbage collection threshold
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### OMP_NUM_THREADS

**Purpose:** Number of OpenMP threads for CPU operations.

**Default:** Number of CPU cores

**Usage:**
```bash
export OMP_NUM_THREADS=8
```

**Recommendation:** Set to 4-8 for training

## Remote Storage

### Google Cloud Storage

#### GOOGLE_APPLICATION_CREDENTIALS

**Purpose:** Path to GCP service account JSON file.

**Usage:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

#### GOOGLE_APPLICATION_CREDENTIALS_JSON

**Purpose:** GCP credentials as raw JSON string (alternative to file).

**Usage:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type": "service_account", ...}'
```

### AWS S3

#### AWS_ACCESS_KEY_ID

**Purpose:** AWS access key for S3.

**Usage:**
```bash
export AWS_ACCESS_KEY_ID=your_key
```

#### AWS_SECRET_ACCESS_KEY

**Purpose:** AWS secret key for S3.

**Usage:**
```bash
export AWS_SECRET_ACCESS_KEY=your_secret
```

#### AWS_DEFAULT_REGION

**Purpose:** Default AWS region.

**Usage:**
```bash
export AWS_DEFAULT_REGION=us-west-2
```

### Weka (AI2 Internal)

#### WEKA_ENDPOINT_URL

**Purpose:** Weka storage endpoint URL.

**Usage:**
```bash
export WEKA_ENDPOINT_URL="https://weka-aus.beaker.org:9000"
```

#### WEKA_PROFILE

**Purpose:** AWS profile name for Weka access.

**Usage:**
```bash
export WEKA_PROFILE="weka"
```

#### AWS_CREDENTIALS

**Purpose:** AWS credentials JSON for Weka access.

**Usage:**
```bash
export AWS_CREDENTIALS='{"aws_access_key_id": "...", "aws_secret_access_key": "..."}'
```

## AI2-Specific

### OLMO_SHARED_FS

**Purpose:** Indicate if using shared filesystem (multi-node).

**Values:** `1` (shared) or `0` (local)

**Usage:**
```bash
export OLMO_SHARED_FS=1  # For remote storage or shared FS
```

**When to set:**
- Multi-node training
- Using remote storage (GCS, S3, Weka)
- Not needed for single-node local training

### BEAKER_EXPERIMENT_ID

**Purpose:** Beaker experiment ID (set automatically by Beaker).

**Usage:** Automatically set, don't manually set

## OpenAI API

### OPENAI_API_KEY

**Purpose:** OpenAI API key for GPT-based evaluations.

**Usage:**
```bash
export OPENAI_API_KEY=sk-your_key_here
```

**When needed:**
- Dense caption evaluation with GPT-4
- Custom evaluations using GPT models

## Development

### PYTHONPATH

**Purpose:** Python module search path.

**Usage:**
```bash
export PYTHONPATH=/path/to/molmo:$PYTHONPATH
```

**When needed:**
- Running scripts outside repo root
- Development without installing package

### CUDA_LAUNCH_BLOCKING

**Purpose:** Synchronous CUDA operations for debugging.

**Values:** `1` (blocking) or `0` (async)

**Usage:**
```bash
export CUDA_LAUNCH_BLOCKING=1  # For debugging
```

**Note:** Significantly slower, only use for debugging

## Example Configurations

### Training Configuration

```bash
#!/bin/bash
# Setup for training

# Data directories
export MOLMO_DATA_DIR=/data/molmo
export HF_HOME=/data/huggingface_cache

# Offline mode (after data downloaded)
export HF_DATASETS_OFFLINE=1

# W&B logging
export WANDB_API_KEY=your_key
export WANDB_ENTITY=my-team
export WANDB_PROJECT=molmo-training

# PyTorch settings
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Distributed training (if multi-node)
export NCCL_TIMEOUT_MINUTES=30

# Shared filesystem flag
export OLMO_SHARED_FS=1
```

### Inference Configuration

```bash
#!/bin/bash
# Setup for inference

# Model cache
export HF_HOME=/cache/huggingface

# Use cached models only
export HF_DATASETS_OFFLINE=1

# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Access token for gated models
export HF_ACCESS_TOKEN=your_token
```

### Development Configuration

```bash
#!/bin/bash
# Setup for development

# Local data
export MOLMO_DATA_DIR=./data
export HF_HOME=./cache

# Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# Debugging
export CUDA_LAUNCH_BLOCKING=1  # When debugging CUDA errors

# Disable W&B
export WANDB_MODE=disabled
```

### Beaker Configuration

```bash
#!/bin/bash
# Setup for Beaker jobs

# Data on Weka
export MOLMO_DATA_DIR=/weka/oe-training-default/mm-olmo

# Offline mode
export HF_DATASETS_OFFLINE=1

# Shared filesystem
export OLMO_SHARED_FS=1

# PyTorch settings
export OMP_NUM_THREADS=8

# NCCL timeout for remote storage
export NCCL_TIMEOUT_MINUTES=30
```

## Permanent Setup

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Molmo Configuration
export MOLMO_DATA_DIR=$HOME/data/molmo
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_OFFLINE=1

# W&B (if always used)
export WANDB_API_KEY=your_key
export WANDB_ENTITY=your-entity

# PyTorch
export OMP_NUM_THREADS=8
```

Then reload:
```bash
source ~/.bashrc
```

## Verification

Check all environment variables:

```bash
env | grep -E "(MOLMO|HF_|WANDB|CUDA|NCCL|OMP)" | sort
```

## See Also

- [Installation Guide](../guides/installation.md)
- [Training Guide](../guides/training_guide.md)
- [Troubleshooting](troubleshooting.md)

