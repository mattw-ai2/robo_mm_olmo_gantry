# Model Architectures

This document details the specific model architectures available in Molmo.

## Molmo (Standard)

The standard Molmo architecture for image-text understanding.

### Configuration

```python
@dataclass
class MolmoConfig(BaseModelConfig):
    llm: LlmConfig                           # Language model config
    vision_backbone: MolmoVisionBackboneConfig  # Vision encoder config
    data_formatter: DataFormatter             # Task formatting
    mm_preprocessor: MolmoPreprocessorConfig  # Input preprocessing
    bi_directional_attn: Optional[str] = None # Bidirectional attention mode
```

### Architecture Components

**Vision Processing:**
- Vision backbone (CLIP/SigLIP/DINO)
- Patch extraction and encoding
- Learnable projection to LLM dimension
- Optional attention pooling
- Position embeddings

**Language Processing:**
- Transformer-based LLM
- Multimodal attention mechanism
- Autoregressive text generation
- Special tokens for images

**Integration:**
- Image tokens inserted into text sequence
- Causal attention for text tokens
- Optional bidirectional attention for image tokens
- Joint training of vision and language components

### Model Sizes

Common configurations:

**Molmo-1B:**
- Vision: CLIP ViT-L/14
- LLM: 1B parameters (MoE or dense)
- Image resolution: 336×336
- Context length: 2048

**Molmo-7B:**
- Vision: CLIP ViT-L/14 or SigLIP
- LLM: 7B parameters
- Image resolution: 336×336 or higher
- Context length: 4096

**Molmo-72B:**
- Vision: Enhanced vision encoder
- LLM: 72B parameters
- Image resolution: Up to 1024×1024
- Context length: 4096

### Forward Pass

```python
def forward(
    self,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    image_masks: Optional[torch.Tensor] = None,
    image_input_idx: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> OLMoOutput:
    """
    Args:
        input_ids: [batch_size, seq_len] - Token IDs
        images: [batch_size, n_images, C, H, W] - Input images
        image_masks: [batch_size, n_images, n_patches] - Valid patches
        image_input_idx: [batch_size, n_images] - Image positions in sequence
        labels: [batch_size, seq_len] - Target tokens for training
    
    Returns:
        OLMoOutput with logits and optional loss
    """
```

### Generation

```python
def generate(
    self,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    **kwargs
) -> OLMoGenerateOutput:
    """Generate text autoregressively."""
```

### Training Modes

1. **Dense Caption Pretraining:**
   - Train on image captioning data
   - Unfreeze vision encoder gradually
   - Large batch sizes (4k-8k images)

2. **Multitask Finetuning:**
   - Mix of captioning, VQA, pointing tasks
   - Balanced task sampling
   - Task-specific prompting

3. **Instruction Tuning:**
   - High-quality instruction-following data
   - Specific output formats (JSON, points, etc.)

## VideoOlmo

Extended architecture for video understanding.

### Configuration

```python
@dataclass
class VideoOlmoConfig(MolmoConfig):
    # Inherits from MolmoConfig with video-specific extensions
    max_frames: int = 16              # Maximum frames per video
    frame_sampling: str = "uniform"   # Frame sampling strategy
```

### Video Processing Pipeline

1. **Frame Extraction:**
   - Uniform sampling or adaptive sampling
   - Configurable frame count (4-64 frames)
   - Temporal position encoding

2. **Per-Frame Encoding:**
   - Each frame encoded independently by vision backbone
   - Shared parameters across frames

3. **Temporal Modeling:**
   - Frames arranged in sequence
   - Temporal attention in LLM
   - Optional temporal aggregation

4. **Generation:**
   - Standard autoregressive generation
   - Context includes all frame features

### Memory Optimization

- **Gradient Checkpointing:** For frame encoding
- **Frame Batching:** Process frames in batches
- **Resolution Reduction:** Lower resolution for many frames
- **Query-based Selection:** Select relevant frames dynamically

### Supported Tasks

- Video captioning
- Video question answering
- Temporal reasoning
- Action recognition
- Video summarization

## HeMolmo (Hierarchical Encoding)

Efficient variant using hierarchical token selection.

### Configuration

```python
@dataclass
class HeMolmoConfig(BaseModelConfig):
    llm: LlmConfig
    vision_backbone: MolmoVisionBackboneConfig
    token_scorer: TokenScorerConfig       # Token selection network
    selection_ratio: float = 0.25         # Fraction of tokens to keep
```

### Key Innovation

Instead of processing all image tokens, HeMolmo:

1. **Initial Encoding:** Encode all image patches
2. **Token Scoring:** Learn which tokens are important
3. **Selection:** Keep only top-k tokens
4. **Processing:** Process selected tokens in LLM

This reduces computation by 50-75% with minimal accuracy loss.

### Token Selection

```python
class TokenSelector(nn.Module):
    def forward(
        self,
        image_features: torch.Tensor,  # [B, N, D]
        query: Optional[torch.Tensor] = None  # Optional query
    ) -> SelectionOutput:
        """
        Select important tokens based on learned scoring.
        
        Returns:
            selected_features: [B, K, D]  # K < N
            selection_indices: [B, K]
        """
```

### Benefits

- **Speed:** 2-3x faster inference
- **Memory:** 50% less GPU memory
- **Quality:** Minimal accuracy degradation
- **Flexibility:** Adjustable selection ratio

### Training

Two-stage training:
1. Train standard Molmo first
2. Add token selector and finetune
3. Optional: Jointly train with selection loss

## MolmoE (Mixture-of-Experts)

Efficient model using sparse MoE architecture.

### Architecture

- **Dense Vision Encoder:** Standard vision backbone
- **Sparse LLM:** Mixture-of-experts transformer
- **Routing:** Learned expert routing per token

### Configuration

```python
# LLM config with MoE
llm_config = LlmConfig(
    block_type="olmoe",          # Use MoE blocks
    n_experts=8,                 # Number of experts
    experts_per_token=2,         # Active experts per token
    ...
)
```

### Benefits

- **Parameter Efficiency:** Large capacity, small active parameters
- **Speed:** Faster inference than dense models
- **Quality:** Competitive with larger dense models

### Trade-offs

- **Training Complexity:** Load balancing required
- **Memory:** All experts in memory
- **Hardware:** Benefits from specialized hardware

## Model Comparison

| Model | Parameters | Active | Speed | Memory | Quality |
|-------|-----------|---------|-------|---------|---------|
| Molmo-1B | 1B | 1B | Fast | Low | Good |
| Molmo-7B | 7B | 7B | Medium | Medium | Excellent |
| Molmo-72B | 72B | 72B | Slow | High | Best |
| MolmoE-1B | 7B | 1B | Fast | Medium | Very Good |
| HeMolmo-7B | 7B | 7B | Fast | Low | Excellent |

## Choosing a Model

**For Research:**
- Molmo-7B: Best balance of quality and efficiency
- Molmo-72B: Best quality, resource intensive

**For Production:**
- MolmoE-1B: Efficient inference, good quality
- HeMolmo-7B: Fast inference with high quality

**For Resource-Constrained:**
- Molmo-1B: Smallest footprint
- HeMolmo variants: Memory-efficient

**For Video:**
- VideoOlmo: Specialized for video understanding

## Implementation Details

### Initialization

Models can be initialized from:
1. **Pretrained LLM + Vision Encoder:** Common starting point
2. **Checkpoint:** Load full Molmo checkpoint
3. **Random:** Initialize from scratch (not recommended)

### Fine-tuning

Best practices:
- Lower learning rate than pretraining (1e-5 to 1e-6)
- Freeze vision encoder initially
- Use task-specific data mixtures
- Monitor overfitting on small datasets

### Inference Optimization

- **Batch Processing:** Process multiple examples together
- **KV Caching:** Cache attention keys/values
- **Quantization:** INT8 or FP16 inference
- **vLLM Integration:** Use vLLM for production serving

## Next Steps

- [Vision Backbone](vision_backbone.md) - Details on vision encoders
- [LLM Components](llm_components.md) - Language model internals
- [Training Guide](../guides/training_guide.md) - How to train models

