# Architecture Overview

This document provides a high-level overview of the Molmo architecture and design principles.

## System Architecture

Molmo is built on a modular architecture that separates concerns into distinct components:

```
┌─────────────────────────────────────────────────────────────┐
│                        Molmo System                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Vision     │───▶│  Multimodal  │───▶│   Language   │  │
│  │   Encoder    │    │   Connector  │    │    Model     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│        │                     │                    │          │
│        │                     │                    │          │
│        ▼                     ▼                    ▼          │
│  Image Features      Fused Features      Text Tokens        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Vision Encoder

The vision encoder processes images and videos into feature representations. Molmo supports multiple vision backbones:

- **CLIP** (OpenAI): Strong general-purpose vision encoder
- **SigLIP** (Google): Improved training with sigmoid loss
- **DINOv2** (Meta): Self-supervised vision features
- **MetaCLIP**: CLIP trained on filtered data

**Key Features:**
- Patch-based processing (e.g., 14×14 patches)
- Multi-resolution support
- Efficient image pooling strategies
- Learnable vision-to-text projection

**Location:** `olmo/nn/image_vit.py`, `olmo/nn/vision_backbone.py`

### 2. Multimodal Connector

The connector module bridges vision and language representations:

- **Image Projection:** Projects vision features to language dimension
- **Positional Encoding:** Maintains spatial information
- **Attention Pooling:** Selective feature aggregation
- **Token Fusion:** Integrates image and text tokens

**Strategies:**
- Linear projection
- MLP projection with residual connections
- Attention-based pooling (2D or 1D)
- C-abstractor style aggregation

**Location:** `olmo/nn/vision_backbone.py`

### 3. Language Model

The LLM backbone generates text conditioned on multimodal inputs. Molmo supports:

- **OLMo:** AI2's open language model
- **Qwen2:** Alibaba's efficient LLM
- **Llama:** Meta's Llama architecture
- **OLMoE:** Mixture-of-experts variant

**Key Features:**
- Transformer-based architecture
- Rotary position embeddings (RoPE)
- Grouped-query attention (GQA)
- SwiGLU activation functions
- Layer normalization variants (RMS, standard)

**Location:** `olmo/nn/llm.py`

## Model Variants

### Standard Molmo

The base architecture for image-text tasks:

```python
class Molmo(ModelBase):
    vision_backbone: MolmoVisionBackbone  # Vision encoder + projection
    llm: Llm                                # Language model
    mm_preprocessor: MolmoPreprocessor     # Input preprocessing
```

**Capabilities:**
- Image captioning
- Visual question answering
- Object pointing
- Counting
- Document understanding

**Location:** `olmo/models/molmo/molmo.py`

### VideoOlmo

Extended architecture for video understanding:

```python
class VideoOlmo(ModelBase):
    # Processes multiple frames with temporal attention
    # Handles variable-length video sequences
```

**Additional Features:**
- Frame sampling strategies
- Temporal encoding
- Query-based frame selection
- Memory-efficient video processing

**Location:** `olmo/models/video_olmo/video_olmo.py`

### HeMolmo

Hierarchical encoding variant for efficiency:

```python
class HeMolmo(ModelBase):
    # Uses hierarchical token selection
    # Reduces computational cost for high-resolution images
```

**Key Innovation:**
- Dynamic token selection
- Multi-scale feature processing
- Reduced memory footprint

**Location:** `olmo/models/he_molmo/he_molmo.py`

## Data Flow

### Training

1. **Data Loading:**
   - Datasets loaded from HuggingFace, local files, or remote storage
   - Dataset mixtures with configurable sampling weights
   - Efficient iterators with prefetching

2. **Preprocessing:**
   - Image resizing and cropping
   - Text tokenization
   - Format conversion (e.g., question-answer to prompt format)
   - Data augmentation (optional)

3. **Collation:**
   - Batch assembly with padding
   - Attention mask generation
   - Image position tracking

4. **Forward Pass:**
   - Vision encoding
   - Feature projection
   - LLM processing
   - Loss computation (cross-entropy, optional auxiliary losses)

5. **Optimization:**
   - Gradient computation
   - Optimizer step (AdamW, Lion)
   - Learning rate scheduling
   - Gradient clipping

6. **Checkpointing:**
   - Periodic model saving
   - Sharded checkpoints for large models
   - Remote storage support (GCS, S3, Weka)

### Inference

1. **Input Processing:**
   - Load and preprocess image/video
   - Format prompt with task-specific templates
   - Generate input tensors

2. **Encoding:**
   - Vision encoding (cached if multiple generations)
   - Position embeddings

3. **Generation:**
   - Autoregressive decoding
   - Beam search or sampling
   - Length constraints
   - Special token handling

4. **Postprocessing:**
   - Decode tokens to text
   - Extract structured outputs (points, bounding boxes)
   - Format results

## Training Infrastructure

### Distributed Training

- **FSDP (Fully Sharded Data Parallel):** For models larger than single GPU memory
- **Activation Checkpointing:** Trades compute for memory
- **Mixed Precision:** FP16/BF16 for efficiency
- **Gradient Accumulation:** Effective larger batch sizes

### Configuration System

Based on OmegaConf with structured configs:

```python
@dataclass
class TrainConfig(BaseConfig):
    model: MolmoConfig           # Model architecture
    data: DataConfig             # Dataset configuration
    optim: OptimizerConfig       # Optimization settings
    fsdp: FSDPConfig            # Distributed training
    save_folder: str             # Checkpoint location
    # ... many more options
```

**Features:**
- Type-safe configuration
- YAML-based config files
- Command-line overrides
- Legacy config migration

### Evaluation Framework

Two-stage evaluation:

1. **Inference Stage:** Generate predictions
   - Batched processing
   - Distributed inference
   - Result caching

2. **Metric Stage:** Compute metrics
   - Task-specific evaluators
   - Standard metrics (VQA accuracy, ANLS, etc.)
   - Custom metrics
   - HTML visualizations

## Key Design Principles

### Modularity

Each component has a clear interface:
- Models implement `ModelBase` with `forward()` and `generate()`
- Datasets implement `Dataset` with `get()` method
- Evaluators implement `Evaluator` with `evaluate()` method

### Flexibility

- Multiple vision encoders supported
- Multiple LLM backbones supported
- Configurable preprocessing pipelines
- Extensible evaluation framework

### Scalability

- Supports training on single GPU to hundreds of GPUs
- Efficient data loading with multiple workers
- Remote storage for datasets and checkpoints
- Checkpointing and resumption

### Reproducibility

- Deterministic data iteration with seeds
- Exact checkpoint recovery
- Comprehensive logging
- Configuration tracking

## File Organization

```
olmo/
├── models/          # Model architectures
│   ├── molmo/      # Standard Molmo
│   ├── video_olmo/ # Video variant
│   └── he_molmo/   # Hierarchical variant
├── nn/             # Neural network components
│   ├── llm.py      # Language model
│   ├── image_vit.py # Vision transformers
│   └── vision_backbone.py # Vision processing
├── data/           # Dataset loaders
├── eval/           # Evaluation framework
├── train/          # Training infrastructure
└── ...
```

## Next Steps

- [Model Architectures](model_architectures.md) - Detailed model documentation
- [Vision Backbone](vision_backbone.md) - Vision encoder details
- [LLM Components](llm_components.md) - Language model details
- [Data Pipeline](data_pipeline.md) - Data processing details

