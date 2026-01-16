# Training Guide

Complete guide to training Molmo models.

## Training Overview

Molmo training typically follows a two-stage process:

1. **Dense Caption Pretraining:** Learn visual-language alignment
2. **Multitask Finetuning:** Specialize on diverse tasks

## Prerequisites

- Completed [installation](installation.md)
- Downloaded datasets (see [data preparation](data_preparation.md))
- Access to GPUs (8x A100 40GB recommended for 7B models)
- Set environment variables:
  ```bash
  export MOLMO_DATA_DIR=/path/to/data
  export HF_HOME=/path/to/huggingface/cache
  export WANDB_API_KEY=your_wandb_key
  ```

## Quick Start Training

### Debug Run (Single GPU)

Test your setup with a minimal run:

```bash
torchrun --nproc-per-node=1 launch_scripts/train_captioner.py debug \
  --save_folder=/path/to/save/folder \
  --save_overwrite
```

### Full Captioning Training (8 GPUs)

Train a dense captioner:

```bash
WANDB_API_KEY=your_key torchrun --nproc-per-node=8 \
  launch_scripts/train_captioner.py qwen2_7b \
  --wandb.name=my-captioner \
  --wandb.entity=my-entity \
  --wandb.project=molmo \
  --save_folder=/path/to/save/folder
```

### Multitask Training

Fine-tune on multiple tasks:

```bash
WANDB_API_KEY=your_key torchrun --nproc-per-node=8 \
  launch_scripts/train_multitask_model.py 3.2-synthetic \
  /path/to/captioner/checkpoint \
  --wandb.name=my-multitask \
  --wandb.entity=my-entity \
  --wandb.project=molmo \
  --save_folder=/path/to/save/folder
```

## Stage 1: Dense Caption Pretraining

### Goal

Learn strong visual-language alignment through dense image captioning.

### Dataset

Primarily uses PixMo-Cap dataset with detailed image descriptions.

### Configuration

```bash
torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b \
  --vision_backbone=openai \           # Vision encoder
  --save_folder=./checkpoints \
  --wandb.name=caption-qwen2-7b \
  --data.num_workers=4 \
  --optimizer.learning_rate=3e-6 \
  --max_steps=15000 \
  --save_interval=1000
```

### Key Parameters

**Model Architecture:**
- `qwen2_7b`: Use Qwen2-7B as base LLM
- `--vision_backbone=openai`: CLIP, SigLIP, or DINO
- `--model.residual_dropout=0.1`: Dropout rate

**Training:**
- `--optimizer.learning_rate=3e-6`: Learning rate
- `--optimizer.weight_decay=0.1`: Weight decay
- `--max_steps=15000`: Training steps
- `--global_train_batch_size=256`: Effective batch size

**Data:**
- `--data.num_workers=4`: Data loading workers
- `--data.pad=to_128`: Padding strategy

### Typical Settings

**7B Model on 8x A100:**
```bash
--global_train_batch_size=256 \
--device_batch_size=4 \
--activation_checkpointing=full \
--fsdp.use_orig_params=False \
--fsdp.sharding_strategy=FULL_SHARD
```

**1B Model on 8x A100:**
```bash
--global_train_batch_size=512 \
--device_batch_size=8 \
--activation_checkpointing=one_in_two
```

### Monitoring

Track these metrics during training:
- **Loss:** Should decrease steadily
- **Throughput:** Tokens/second
- **GPU Memory:** Should be stable
- **Learning Rate:** Follows schedule

### Checkpointing

Checkpoints saved every `save_interval` steps to `save_folder`:
```
checkpoints/
├── step0/
├── step1000/
├── step2000/
├── ...
└── latest/
```

## Stage 2: Multitask Finetuning

### Goal

Specialize model on diverse tasks: VQA, pointing, counting, etc.

### Dataset Mixture

Example mixture `3.2-synthetic`:
- PixMo datasets (captioning, points, counting)
- VQA datasets (VQA2, TextVQA, etc.)
- Academic benchmarks (AI2D, ChartQA, etc.)

### Starting from Captioner

```bash
torchrun --nproc-per-node=8 \
  launch_scripts/train_multitask_model.py 3.2-synthetic \
  /path/to/captioner/checkpoint/step15000 \
  --wandb.name=multitask-v1 \
  --save_folder=./multitask_checkpoints \
  --max_steps=10000 \
  --optimizer.learning_rate=1e-6
```

### Task Balancing

Configure dataset mixing weights in the training script:

```python
datasets = [
    {"name": "pixmo_cap", "weight": 0.3},
    {"name": "pixmo_points", "weight": 0.2},
    {"name": "vqa2", "weight": 0.15},
    {"name": "text_vqa", "weight": 0.15},
    {"name": "chart_qa", "weight": 0.1},
    {"name": "count_qa", "weight": 0.1},
]
```

### Typical Settings

**Multitask 7B on 8x A100:**
```bash
--global_train_batch_size=128 \
--device_batch_size=2 \
--optimizer.learning_rate=1e-6 \
--max_steps=10000
```

### Evaluation During Training

The training script evaluates on validation sets:
```bash
--evaluations.0.data.dataset=vqa2:validation \
--evaluations.0.eval_interval=1000
```

## Distributed Training

### Single Node (8 GPUs)

```bash
torchrun --nproc-per-node=8 launch_scripts/train_captioner.py ...
```

### Multi-Node (2 nodes, 8 GPUs each)

**Node 0 (master):**
```bash
torchrun \
  --nproc-per-node=8 \
  --nnodes=2 \
  --node-rank=0 \
  --master-addr=<master-ip> \
  --master-port=29500 \
  launch_scripts/train_captioner.py ...
```

**Node 1:**
```bash
torchrun \
  --nproc-per-node=8 \
  --nnodes=2 \
  --node-rank=1 \
  --master-addr=<master-ip> \
  --master-port=29500 \
  launch_scripts/train_captioner.py ...
```

### FSDP Configuration

For large models, use Fully Sharded Data Parallel:

```bash
--fsdp=True \
--fsdp.sharding_strategy=FULL_SHARD \
--fsdp.use_orig_params=False \
--fsdp.precision=bf16
```

**Sharding Strategies:**
- `FULL_SHARD`: Shard parameters, gradients, and optimizer states
- `SHARD_GRAD_OP`: Shard gradients and optimizer states only
- `NO_SHARD`: Standard DDP (not recommended for large models)

## Advanced Configuration

### Learning Rate Scheduling

```bash
--scheduler.name=linear_with_warmup \
--scheduler.units=steps \
--scheduler.t_warmup=1000 \
--scheduler.t_max=15000
```

**Available Schedulers:**
- `linear_with_warmup`: Linear decay after warmup
- `cosine_with_warmup`: Cosine decay after warmup
- `constant`: Fixed learning rate
- `inverse_sqrt_with_warmup`: Inverse square root decay

### Gradient Clipping

```bash
--max_grad_norm=1.0  # Clip gradients to this norm
```

### Activation Checkpointing

Trade compute for memory:

```bash
--activation_checkpointing=full      # Maximum memory savings
--activation_checkpointing=one_in_two  # Checkpoint every other layer
```

### Compilation (PyTorch 2.x)

Speed up training with torch.compile:

```bash
--compiler.mode=default               # Enable compilation
--compiler.dynamic=False              # Static shapes for best performance
```

**Compilation Modes:**
- `default`: Balanced speed and compile time
- `reduce-overhead`: Maximum performance
- `max-autotune-no-cudagraphs`: Extensive optimization

### Mixed Precision

```bash
--fsdp.precision=bf16  # Use BFloat16 (recommended)
--fsdp.precision=fp16  # Use Float16 (may be unstable)
--fsdp.precision=pure_bf16  # Pure BF16 (no FP32 master weights)
```

## Custom Datasets

### Adding a New Dataset

1. **Create dataset class:**
   ```python
   from olmo.data.dataset import DatasetBase
   
   class MyDataset(DatasetBase):
       def load(self):
           # Load your data
           return data
           
       def get(self, idx, rng):
           example = self.data[idx]
           return {
               "image": example["image"],
               "messages": [
                   {"role": "user", "content": example["question"]},
                   {"role": "assistant", "content": example["answer"]},
               ]
           }
   ```

2. **Register in training script:**
   ```python
   from my_dataset import MyDataset
   
   dataset = MyDataset(split="train")
   ```

3. **Add to mixture:**
   ```python
   {"name": "my_dataset", "weight": 0.15}
   ```

See [Custom Datasets](../datasets/custom_datasets.md) for details.

## Resuming Training

Training automatically resumes from the latest checkpoint:

```bash
# Same command as original training
torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b \
  --save_folder=/same/folder \
  --allow_resume=True  # Default is True
```

The trainer will:
1. Find latest checkpoint
2. Restore model, optimizer, scheduler state
3. Continue from next step
4. Create new W&B run (linked to previous)

## Hyperparameter Tuning

### Key Hyperparameters

**Learning Rate:**
- Start: 3e-6 to 1e-5 for pretraining
- Finetuning: 1e-6 to 5e-6
- Rule: Larger models need smaller LR

**Batch Size:**
- Effective batch size: 256-1024
- Per-device batch size: As large as memory allows
- Gradient accumulation: To reach effective batch size

**Warmup:**
- 5-10% of total steps
- Helps stabilize early training

**Weight Decay:**
- 0.1 is typical
- May help prevent overfitting

### Experiment Tracking

Use W&B sweeps for hyperparameter search:

```yaml
# sweep.yaml
program: launch_scripts/train_captioner.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  optimizer.learning_rate:
    min: 1e-6
    max: 1e-5
  optimizer.weight_decay:
    values: [0.0, 0.1, 0.2]
```

```bash
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

## Troubleshooting

### Out of Memory

**Solutions:**
1. Reduce `device_batch_size`
2. Enable/increase `activation_checkpointing`
3. Use `FULL_SHARD` FSDP
4. Reduce image resolution
5. Use gradient accumulation

### Slow Training

**Solutions:**
1. Increase `num_workers` for data loading
2. Use faster storage (local SSD vs network)
3. Enable `torch.compile`
4. Use `activation_checkpointing=one_in_two` instead of `full`
5. Increase batch size

### NaN Loss

**Solutions:**
1. Reduce learning rate
2. Increase warmup steps
3. Use BF16 instead of FP16
4. Enable gradient clipping
5. Check data for issues

### Training Instability

**Solutions:**
1. Longer warmup
2. Lower learning rate
3. Gradient clipping
4. Check for data quality issues
5. Use BF16 mixed precision

## Best Practices

1. **Start Small:** Debug with small model/data first
2. **Monitor Metrics:** Watch loss, throughput, memory
3. **Save Often:** Regular checkpoints prevent data loss
4. **Validate Early:** Catch issues before long training
5. **Document:** Keep notes on experiments
6. **Version Control:** Track code and config changes

## Example Training Pipelines

### Research Workflow

```bash
# 1. Debug run
torchrun --nproc-per-node=1 launch_scripts/train_captioner.py debug

# 2. Small-scale test
torchrun --nproc-per-node=4 launch_scripts/train_captioner.py qwen2_7b \
  --max_steps=1000

# 3. Full training
torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b \
  --max_steps=15000

# 4. Multitask
torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py ...
```

### Production Workflow

```bash
# 1. Pretrain captioner (2 days, 8xA100)
torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b \
  --save_folder=gs://bucket/captioner \
  --max_steps=15000

# 2. Multitask finetune (1 day, 8xA100)
torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py 3.2 \
  gs://bucket/captioner/step15000 \
  --save_folder=gs://bucket/multitask

# 3. Evaluate
torchrun --nproc-per-node=8 launch_scripts/eval_downstream.py \
  gs://bucket/multitask/step10000 \
  high-res --high_res --fsdp

# 4. Convert to HF format
python scripts/hf_molmo/convert_molmo_to_hf.py \
  gs://bucket/multitask/step10000 \
  ./hf_model

# 5. Upload to HF Hub
huggingface-cli upload my-org/my-molmo ./hf_model
```

## Next Steps

- **[Evaluation Guide](evaluation_guide.md)** - Evaluate trained models
- **[Distributed Training](distributed_training.md)** - Multi-node training
- **[Data Preparation](data_preparation.md)** - Prepare custom datasets
- **[Configuration Reference](../reference/configuration_reference.md)** - All config options

