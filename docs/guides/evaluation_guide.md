# Evaluation Guide

Complete guide to evaluating Molmo models on downstream tasks.

## Overview

Molmo supports evaluation on 20+ benchmarks including:
- Visual Question Answering (VQA2, TextVQA, etc.)
- Document Understanding (DocQA, ChartQA)
- Mathematical Reasoning (MathVista)
- Video Understanding (VideoMME, MLVU)
- Visual Grounding (RefCOCO, pointing tasks)
- Counting (TallyQA, CountBench)

## Quick Start

### Evaluate on Single Task

```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 \
  text_vqa \
  --save_to_checkpoint_dir
```

### Evaluate on Multiple Tasks

```bash
# Standard eval set (11 tasks, low resolution)
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 \
  low-res \
  --save_to_checkpoint_dir

# High resolution eval (slower but better)
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 \
  high-res \
  --save_to_checkpoint_dir \
  --high_res \
  --fsdp \
  --device_batch_size=2
```

## Evaluation Process

Evaluation happens in two stages:

### Stage 1: Generate Predictions

```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  /path/to/checkpoint \
  text_vqa \
  --save_to_checkpoint_dir
```

**Output:**
- Predictions saved to `checkpoint_dir/predictions-text_vqa-validation/`
- Includes `predictions.json` and metrics

### Stage 2: Compute Metrics

Metrics are computed automatically after generation. To recompute:

```bash
python launch_scripts/eval_downstream.py \
  /path/to/checkpoint \
  text_vqa \
  --metrics_only \
  --predictions_dir=path/to/predictions
```

## Supported Tasks

### Visual Question Answering

**VQA2:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 vqa2
```

**TextVQA:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 text_vqa --high_res --fsdp
```

**DocQA:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 doc_qa --high_res --fsdp
```

**Metrics:** VQA accuracy, ANLS

### Chart/Figure Understanding

**ChartQA:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 chart_qa --high_res --fsdp
```

**AI2D:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 ai2d
```

**Metrics:** Exact match, relaxed accuracy

### Mathematical Reasoning

**MathVista:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 math_vista --high_res --fsdp
```

**Metrics:** Accuracy on different problem types

### Video Understanding

**VideoMME:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_video.py \
  VideoOlmo-7B \
  video_mme \
  --max_frames=16
```

**MLVU:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_video.py \
  VideoOlmo-7B \
  mlvu_gen
```

**Metrics:** Video QA accuracy, temporal reasoning

### Counting

**TallyQA:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 tally_qa --high_res --fsdp
```

**CountBench:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 count_bench
```

**Metrics:** Counting accuracy, RMSE

### Pointing/Grounding

**PixMo Points:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 pixmo_points:validation --high_res --fsdp
```

**Metrics:** Pointing accuracy (within threshold)

## Evaluation Options

### Resolution

**Standard Resolution (336×336):**
```bash
# Faster, less memory
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 text_vqa
```

**High Resolution (768×768):**
```bash
# Better quality, requires FSDP
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 text_vqa --high_res --fsdp --device_batch_size=2
```

### FSDP for Evaluation

Enable FSDP for large models or high resolution:

```bash
--fsdp \
--device_batch_size=2  # Reduce batch size with FSDP
```

### Caching

Evaluation results are cached. To overwrite:

```bash
--overwrite  # Regenerate predictions and metrics
```

To skip if already evaluated:

```bash
--skip_if_metrics_cached  # Skip tasks with existing metrics
```

### Output Location

**Save to checkpoint directory (default):**
```bash
--save_to_checkpoint_dir
```

**Save to custom location:**
```bash
--save_folder=/path/to/results
```

## Evaluation Sets

### Predefined Sets

**low-res:** 11 core tasks at standard resolution
- VQA2, TextVQA, DocQA, AI2D, ChartQA
- OkVQA, ScienceQA, TallyQA
- MMMU, MathVista, RealWorldQA

**high-res:** Same tasks at high resolution

**low-res-test:** Test set versions

**high-res-test:** High res test sets

### Custom Sets

Create custom evaluation sets in `launch_scripts/eval_downstream.py`:

```python
EVAL_SETS = {
    "my_eval": [
        "vqa2",
        "text_vqa:validation",
        "chart_qa",
    ]
}
```

## Metrics

### Automatic Metrics

Computed automatically for each task:

**VQA Tasks:**
- VQA accuracy (3/3 correct = 100%, 2/3 = 67%, etc.)
- Exact match

**Text Tasks:**
- ANLS (Average Normalized Levenshtein Similarity)
- Relaxed correctness

**Counting:**
- Exact count accuracy
- RMSE (Root Mean Square Error)

**Pointing:**
- Accuracy within N pixels (default: 20px)
- Average distance

**Multiple Choice:**
- Accuracy
- Per-category breakdown

### Custom Metrics

Implement custom evaluators in `olmo/eval/evaluators.py`:

```python
class MyCustomEval(Evaluator):
    def evaluate(self, predictions: List[Dict], ground_truth: List[Dict]):
        # Compute your metric
        accuracy = compute_accuracy(predictions, ground_truth)
        return {"accuracy": accuracy}
```

## Analyzing Results

### Viewing Metrics

Metrics are saved to:
```
checkpoint_dir/
└── predictions-task-split/
    ├── predictions.json
    ├── metrics.json
    └── visualization.html (if available)
```

**Load metrics:**
```python
import json

with open("metrics.json") as f:
    metrics = json.load(f)
    
print(f"Accuracy: {metrics['accuracy']:.2f}")
```

### Visualization

Some tasks generate HTML visualizations:

```bash
# Open in browser
firefox checkpoint_dir/predictions-text_vqa-validation/visualization.html
```

Visualizations show:
- Input images
- Questions
- Predictions
- Ground truth
- Correctness

### Comparing Models

```python
import pandas as pd

results = {
    "Molmo-1B": {"vqa2": 72.5, "text_vqa": 58.3},
    "Molmo-7B": {"vqa2": 77.8, "text_vqa": 65.2},
    "Molmo-72B": {"vqa2": 82.3, "text_vqa": 72.1},
}

df = pd.DataFrame(results).T
print(df)
```

## Test Set Evaluation

### Generating Test Predictions

```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
  Molmo-7B-D-0924 \
  vqa2:test \
  --save_to_checkpoint_dir
```

### Formatting for Submission

Some benchmarks require specific formats:

```bash
python scripts/build_submission_file.py \
  --predictions=predictions.json \
  --task=vqa2 \
  --output=submission.json
```

### Submitting to Servers

- **VQA2:** https://eval.ai/web/challenges/challenge-page/830/overview
- **TextVQA:** https://textvqa.org/challenge
- **DocQA:** https://rrc.cvc.uab.es/
- **MMMU:** https://mmmu-benchmark.github.io/

## Dense Caption Evaluation

### Generating Captions

```bash
torchrun --nproc-per-node 8 launch_scripts/eval.py \
  --task dense_caption_eval \
  /path/to/checkpoint
```

### GPT-4 Based Evaluation

```bash
export OPENAI_API_KEY=your_key

python scripts/gpt_dense_caption_eval.py \
  predictions.json \
  --sample 1500 \
  --metrics all
```

**Metrics:**
- Caption quality (GPT-4 rated)
- Detail level
- Accuracy
- Coverage

### Loss-Based Evaluation

```bash
torchrun --nproc-per-node 8 launch_scripts/eval_captioner.py \
  /path/to/checkpoint \
  --loss \
  --seq_len=2048 \
  --task=pixmo_cap \
  --split=validation
```

## Video Evaluation

### Video-Specific Tasks

```bash
# VideoMME
torchrun --nproc-per-node 8 launch_scripts/eval_video.py \
  VideoOlmo-7B video_mme --max_frames=16

# MLVU
torchrun --nproc-per-node 8 launch_scripts/eval_video.py \
  VideoOlmo-7B mlvu_gen

# TempCompass
torchrun --nproc-per-node 8 launch_scripts/eval_video.py \
  VideoOlmo-7B temp_compass
```

### Frame Configuration

```bash
--max_frames=16  # Number of frames to sample
--frame_sampling=uniform  # uniform, random, or adaptive
```

## Batch Evaluation

### Evaluate All Tasks

```bash
#!/bin/bash
tasks=("vqa2" "text_vqa" "doc_qa" "ai2d" "chart_qa")

for task in "${tasks[@]}"; do
    echo "Evaluating $task..."
    torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
        Molmo-7B-D-0924 "$task" --save_to_checkpoint_dir
done
```

### Parallel Evaluation

Evaluate multiple tasks in parallel on different GPUs:

```bash
# Terminal 1 (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
    launch_scripts/eval_downstream.py Molmo-7B-D-0924 vqa2

# Terminal 2 (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 \
    launch_scripts/eval_downstream.py Molmo-7B-D-0924 text_vqa
```

## Troubleshooting

### Out of Memory

**Solutions:**
1. Reduce `device_batch_size`
2. Use lower resolution
3. Enable FSDP: `--fsdp`
4. Reduce `max_frames` for video

### Slow Evaluation

**Solutions:**
1. Increase `device_batch_size`
2. Reduce precision: `--precision=fp16`
3. Use lower resolution for development
4. Cache predictions

### Incorrect Metrics

**Check:**
1. Correct split used (validation vs test)
2. Predictions format matches expected
3. Ground truth loaded correctly
4. Task-specific preprocessing applied

## Best Practices

1. **Use Validation First:** Debug on validation before test
2. **Cache Results:** Don't regenerate unnecessarily
3. **Document Settings:** Record resolution, batch size, etc.
4. **Compare Fairly:** Use same settings across models
5. **Check Visualizations:** Verify predictions make sense
6. **Test Set Caution:** Only evaluate test set once for final results

## Evaluation with VLMEvalKit

Alternative evaluation using VLMEvalKit:

```bash
# Install VLMEvalKit
pip install vlmeval

# Evaluate
python -m vlmeval.run \
    --model Molmo-7B-D-0924 \
    --dataset VQAv2 TextVQA DocVQA \
    --work_dir ./eval_results
```

See [VLMEvalKit docs](https://github.com/open-compass/VLMEvalKit) for details.

## Next Steps

- **[Training Guide](training_guide.md)** - Train models
- **[Deployment Guide](deployment_guide.md)** - Deploy to production
- **[API Reference](../api/evaluation.md)** - Evaluation API
- **[Troubleshooting](../reference/troubleshooting.md)** - Common issues

