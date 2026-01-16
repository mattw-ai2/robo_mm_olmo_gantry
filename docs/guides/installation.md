# Installation Guide

This guide covers installing Molmo for different use cases.

## Prerequisites

### System Requirements

- **Operating System:** Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python:** 3.10 or higher
- **CUDA:** 11.8 or higher (for GPU support)
- **GPU:** NVIDIA GPU with 16GB+ VRAM recommended for training
- **RAM:** 32GB+ recommended for training
- **Storage:** 100GB+ for datasets and models

### Software Prerequisites

- **Git:** For cloning the repository
- **PyTorch:** 2.3.1 or higher
- **CUDA Toolkit:** Matching your PyTorch version

## Installation Methods

### Method 1: Quick Install (Recommended)

For most users, the quick install is sufficient:

```bash
# Clone the repository
git clone https://github.com/allenai/molmo.git
cd molmo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Molmo with all dependencies
pip install -e .[all]
```

### Method 2: Minimal Install

For inference only:

```bash
git clone https://github.com/allenai/molmo.git
cd molmo

python -m venv venv
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Method 3: Development Install

For contributors and developers:

```bash
git clone https://github.com/allenai/molmo.git
cd molmo

python -m venv venv
source venv/bin/activate

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install with dev dependencies
pip install -e .[dev,train,serve]

# Install pre-commit hooks
pre-commit install
```

### Method 4: Docker Install

Using the provided Dockerfile:

```bash
# Build Docker image
docker build -t molmo:latest -f Dockerfile .

# Run container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v /path/to/data:/data \
  molmo:latest bash
```

## Installing PyTorch

PyTorch installation varies by CUDA version. Visit [pytorch.org](https://pytorch.org) for the latest instructions.

### CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only (Not Recommended)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Environment Setup

### Required Environment Variables

```bash
# Data directory
export MOLMO_DATA_DIR=/path/to/data

# HuggingFace cache
export HF_HOME=/path/to/huggingface/cache
```

### Optional Environment Variables

```bash
# Weights & Biases
export WANDB_API_KEY=your_wandb_key
export WANDB_ENTITY=your_entity
export WANDB_PROJECT=your_project

# OpenAI API (for GPT-based evaluation)
export OPENAI_API_KEY=your_openai_key

# HuggingFace token (for gated models)
export HF_ACCESS_TOKEN=your_hf_token

# Offline mode (use cached datasets)
export HF_DATASETS_OFFLINE=1
```

### Permanent Setup

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Molmo Configuration
export MOLMO_DATA_DIR=$HOME/data/molmo
export HF_HOME=$HOME/.cache/huggingface
export WANDB_API_KEY=your_wandb_key
```

## Installing Optional Dependencies

### For MoE Models (MolmoE)

```bash
pip install git+https://github.com/Muennighoff/megablocks.git@olmoe
```

### For Serving

```bash
pip install vllm
pip install -e .[serve]
```

### For Development

```bash
pip install -e .[dev]

# Linting and formatting
pip install ruff black mypy

# Testing
pip install pytest pytest-cov
```

## Verifying Installation

### Basic Import Test

```python
import torch
from olmo import Molmo
from olmo.models.model_config import ModelConfig

print("✓ Imports successful")
```

### Model Loading Test

```python
from transformers import AutoModelForCausalLM

# Load a small model to verify
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
print("✓ Model loading successful")
```

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run specific test
pytest tests/data/test_preprocessor.py

# With coverage
pytest --cov=olmo tests/
```

## Troubleshooting

### CUDA Out of Memory

**Problem:** GPU runs out of memory during model loading.

**Solutions:**
1. Use a smaller model (e.g., 1B instead of 7B)
2. Reduce batch size
3. Enable gradient checkpointing
4. Use CPU offloading

```python
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatic device placement
    low_cpu_mem_usage=True,
)
```

### Import Errors

**Problem:** `ModuleNotFoundError` when importing.

**Solutions:**
1. Ensure installation: `pip install -e .`
2. Check Python path: `echo $PYTHONPATH`
3. Activate virtual environment: `source venv/bin/activate`
4. Reinstall dependencies: `pip install -e .[all] --force-reinstall`

### CUDA Version Mismatch

**Problem:** PyTorch CUDA version doesn't match system CUDA.

**Solutions:**
1. Check system CUDA: `nvcc --version`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.version.cuda)"`
3. Reinstall PyTorch with matching CUDA version

### Slow Data Loading

**Problem:** Training is slow due to data loading.

**Solutions:**
1. Increase num_workers in DataLoader
2. Store data on fast SSD instead of network drive
3. Use cached datasets with `HF_DATASETS_OFFLINE=1`
4. Preprocess datasets in advance

### Permission Errors

**Problem:** Cannot write to data directories.

**Solutions:**
1. Check directory permissions: `ls -la $MOLMO_DATA_DIR`
2. Create directories with proper permissions:
   ```bash
   mkdir -p $MOLMO_DATA_DIR
   chmod 755 $MOLMO_DATA_DIR
   ```
3. Use a directory you have write access to

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip python3-venv git

# Install CUDA (if not already installed)
# Follow: https://developer.nvidia.com/cuda-downloads

# Continue with standard installation
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Continue with standard installation
# Note: No CUDA support on macOS, CPU only
```

### Windows (WSL2)

```bash
# Install WSL2
wsl --install -d Ubuntu-22.04

# Inside WSL2, follow Ubuntu instructions
# Install CUDA in WSL2:
# https://docs.nvidia.com/cuda/wsl-user-guide/index.html
```

## Cluster-Specific Setup

### Beaker (AI2)

```bash
# On Beaker, use provided Docker image
# Set environment variables in Beaker experiment:
--env MOLMO_DATA_DIR=/weka/oe-training-default/mm-olmo
--env HF_DATASETS_OFFLINE=1
--env OLMO_SHARED_FS=1
```

### SLURM Clusters

```bash
# Load modules
module load cuda/12.1
module load python/3.10

# Create environment in your home directory
python -m venv ~/.venvs/molmo
source ~/.venvs/molmo/bin/activate

# Continue with standard installation
```

## Next Steps

- **[Quick Start](quickstart.md)** - Run your first Molmo model
- **[Training Guide](training_guide.md)** - Start training models
- **[Data Preparation](data_preparation.md)** - Download and prepare datasets
- **[Configuration](configuration.md)** - Configure Molmo for your needs

