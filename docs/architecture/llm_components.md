# LLM Components

This document details the language model components used in Molmo.

## Overview

The LLM (Large Language Model) is the core component that processes multimodal inputs and generates text outputs. Molmo supports multiple LLM architectures with consistent interfaces.

## Supported LLM Architectures

### OLMo

AI2's open language model:

**Features:**
- Fully open training data and code
- Strong performance on language tasks
- Sizes: 1B, 7B, 13B
- Apache 2.0 license

**Configuration:**
```python
llm_config = LlmConfig(
    vocab_size=50280,
    embedding_size=50304,
    d_model=4096,
    n_heads=32,
    n_layers=32,
    mlp_hidden_size=14336,
    activation_type=ActivationType.swiglu,
    block_type="sequential",
)
```

### Qwen2

Alibaba's efficient language model:

**Features:**
- Excellent multilingual support
- Efficient architecture
- Strong reasoning capabilities
- Sizes: 0.5B to 72B

**Advantages:**
- Fast inference
- Good instruction following
- Competitive performance

### Llama

Meta's popular language model:

**Features:**
- Wide adoption
- Strong base performance
- Good fine-tuning behavior
- Sizes: 7B, 13B, 70B

## LLM Architecture

### Core Transformer

```python
class Llm(nn.Module):
    def __init__(self, config: LlmConfig):
        self.embedding = Embedding(config)
        self.blocks = nn.ModuleList([
            OLMoBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = LayerNorm(config)
        self.ff_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
```

### Input Embedding

```python
class Embedding(nn.Module):
    """Token embedding with optional weight tying."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        # returns: [batch_size, seq_len, d_model]
        return self.weight[x]
```

**Features:**
- Learnable token embeddings
- Weight tying with output layer (optional)
- Additional vocab for special tokens (image tokens)

### Transformer Block

```python
class OLMoBlock(nn.Module):
    def __init__(self, config: LlmConfig):
        self.attn_norm = LayerNorm(config)
        self.attention = Attention(config)
        self.ff_norm = LayerNorm(config)
        self.feed_forward = FeedForward(config)
        self.dropout = Dropout(config.residual_dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        h = x + self.dropout(self.attention(
            self.attn_norm(x),
            attention_bias,
            position_ids,
            past_key_value
        ))
        
        # Pre-norm feed-forward
        h = h + self.dropout(self.feed_forward(self.ff_norm(h)))
        
        return h
```

### Attention Mechanism

**Grouped-Query Attention (GQA):**

```python
class Attention(nn.Module):
    def __init__(self, config: LlmConfig):
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads  # < n_heads for GQA
        self.head_dim = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
```

**Key Features:**
- Multi-head attention with configurable heads
- Grouped-query attention for efficiency
- Flash attention support
- Causal masking for autoregressive generation

**Attention Bias:**
```python
# Causal attention: tokens attend only to previous tokens
attention_bias = torch.triu(
    torch.ones(seq_len, seq_len) * float('-inf'),
    diagonal=1
)

# With images: image tokens can attend bidirectionally
# Text tokens attend causally
```

### Position Encodings

**Rotary Position Embeddings (RoPE):**

```python
class RotaryEmbedding(nn.Module):
    """Rotary position embeddings for better length generalization."""
    
    def forward(
        self,
        q: torch.Tensor,  # [B, H, L, D]
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply rotary embeddings to q and k
        cos, sin = self.get_cos_sin(position_ids)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
```

**Benefits:**
- Better length extrapolation
- Relative position information
- No learned position parameters
- Works well with variable lengths

### Feed-Forward Network

**SwiGLU Activation:**

```python
class SwiGLU(nn.Module):
    """Gated linear unit with Swish activation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class FeedForward(nn.Module):
    def __init__(self, config: LlmConfig):
        self.w1 = nn.Linear(config.d_model, config.mlp_hidden_size * 2)
        self.w2 = nn.Linear(config.mlp_hidden_size, config.d_model)
        self.act = SwiGLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))
```

**Typical size:** `mlp_hidden_size = 3.5 * d_model`

### Normalization

**RMS Layer Norm:**

```python
class RMSLayerNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # More efficient than LayerNorm
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
```

**Advantages over LayerNorm:**
- Faster computation (no mean subtraction)
- Similar performance
- Used in Llama, Qwen2

## Block Types

### Sequential Block

Standard transformer block with attention → FFN:

```python
block_type = "sequential"
```

### Llama Block

Llama-style architecture:

```python
block_type = "llama"
# - RMS normalization
# - RoPE position embeddings
# - SwiGLU activation
# - No bias terms
```

### OLMoE Block

Mixture-of-experts variant:

```python
block_type = "olmoe"
# - Sparse FFN layers
# - Expert routing
# - Load balancing
```

## Multimodal Integration

### Image Token Handling

Images are converted to sequences of tokens:

```python
# Image features: [B, n_patches, d_model]
# Inserted into text sequence at special positions

# Example sequence:
# [BOS] <image> <img_0> <img_1> ... <img_N> </image> Describe this image [EOS]
```

### Attention Patterns

**Causal Attention (Text):**
- Text tokens attend only to previous tokens
- Autoregressive generation

**Bidirectional Attention (Images):**
- Image tokens can attend to each other
- Better image understanding
- Configured via `bi_directional_attn`

**Mixed Attention:**
```python
# Image tokens: bidirectional
# Text tokens: causal
# Implemented via attention bias mask
```

## Configuration Options

### Model Size

```python
# Small model (1B parameters)
LlmConfig(
    d_model=2048,
    n_heads=16,
    n_layers=16,
    mlp_hidden_size=5504,
)

# Medium model (7B parameters)
LlmConfig(
    d_model=4096,
    n_heads=32,
    n_layers=32,
    mlp_hidden_size=14336,
)

# Large model (72B parameters)
LlmConfig(
    d_model=8192,
    n_heads=64,
    n_layers=80,
    mlp_hidden_size=28672,
)
```

### Attention Configuration

```python
# Multi-head attention
n_heads = 32
n_kv_heads = 32  # Same as n_heads

# Grouped-query attention (more efficient)
n_heads = 32
n_kv_heads = 8  # 4 query heads per kv head

# Multi-query attention (most efficient)
n_heads = 32
n_kv_heads = 1  # All queries share kv heads
```

### Activation Functions

```python
# SwiGLU (recommended)
activation_type = ActivationType.swiglu

# GELU
activation_type = ActivationType.gelu

# ReLU
activation_type = ActivationType.relu
```

### Dropout

```python
# Embedding dropout
embedding_dropout = 0.1

# Residual dropout (after attention/FFN)
residual_dropout = 0.1

# Attention dropout
attention_dropout = 0.1
```

## Training Considerations

### Initialization

```python
# Load pretrained LLM
llm_config.llm_load_path = "path/to/checkpoint"

# Freeze LLM initially
llm_config.llm_freeze = True

# Unfreeze after warmup
llm_config.llm_unfreeze_step = 1000
```

### Gradient Checkpointing

```python
# Full checkpointing (save memory)
activation_checkpointing = "full"

# Selective checkpointing
activation_checkpointing = "one_in_two"

# No checkpointing (faster but more memory)
activation_checkpointing = None
```

### Mixed Precision

```python
# BF16 (recommended for training)
fsdp_precision = "bf16"

# FP16 (faster but less stable)
fsdp_precision = "fp16"

# Pure BF16 (no FP32 master weights)
fsdp_precision = "pure_bf16"
```

## Inference Optimization

### KV Caching

Cache attention keys and values during generation:

```python
# First forward: compute all KVs
past_key_values = None
output = model(input_ids, past_key_values=past_key_values)

# Subsequent forwards: only new token
past_key_values = output.past_key_values
new_token_output = model(new_token_id, past_key_values=past_key_values)
```

### Quantization

```python
# INT8 quantization
model = model.quantize(bits=8)

# INT4 quantization (more aggressive)
model = model.quantize(bits=4)
```

### Compilation

```python
# PyTorch 2.0 compilation
model = torch.compile(model, mode="default")

# Or max-autotune
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

## Generation Strategies

### Greedy Decoding

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=False,
)
```

### Sampling

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
```

### Beam Search

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    num_beams=5,
    early_stopping=True,
)
```

## Performance Benchmarks

Typical throughput (tokens/second):

| Model | Batch=1 | Batch=8 | Memory (GB) |
|-------|---------|---------|-------------|
| 1B | 150 | 800 | 4 |
| 7B | 40 | 250 | 16 |
| 72B | 5 | 30 | 144 |

*Measured on A100 40GB GPU*

## Best Practices

### For Training
- Use gradient checkpointing for large models
- Start with frozen LLM, unfreeze gradually
- Use BF16 mixed precision
- Monitor gradient norms

### For Inference
- Enable KV caching
- Use appropriate batch size
- Consider quantization for deployment
- Use vLLM for production serving

### For Fine-tuning
- Lower learning rate (1e-5 to 1e-6)
- Smaller batch size than pretraining
- More frequent evaluation
- Watch for overfitting

## Next Steps

- [Data Pipeline](data_pipeline.md) - How data flows through the model
- [Training Guide](../guides/training_guide.md) - Training procedures
- [Deployment Guide](../guides/deployment_guide.md) - Production deployment

