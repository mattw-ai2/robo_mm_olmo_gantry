# Robot Training Guide for robo_mm_olmo

Complete guide for training the robot navigation agent in robo_mm_olmo.

## Table of Contents

1. [Robot Prompt Location and Content](#robot-prompt-location-and-content)
2. [How to Modify the Prompt](#how-to-modify-the-prompt)
3. [Setting Up Robot Training](#setting-up-robot-training)
4. [Configuration Options](#configuration-options)
5. [Training Mixtures](#training-mixtures)
6. [Data Location](#data-location)
7. [Quick Start Examples](#quick-start-examples)

---

## Robot Prompt Location and Content

### Location
The robot prompt is located in:
```
/weka/prior/mattw/robo_mm_olmo/olmo/data/robot_datasets.py
```

**Lines: 1396-1422** (in the `_construct_prompts` method)

### Current Prompt Template

```python
prompt_template = (
    "You are a robot with four cameras, arranged clockwise as front, "
    "right, down, and left. Your goal is {goal}."
)

# Optional context added based on memory settings
if include_scene:
    context_parts.append(f"{scene_str}")  # Scene description
if include_objects:
    context_parts.append(f"You have been close to {objects_str}")  # Objects seen

if context_parts:
    prompt_template += " " + " and ".join(context_parts) + "."

# Main instruction
prompt_template += (
    " Point to a point on the floor to walk towards "
    "or an object to approach in service of your goal. If you have satisfied "
    'your goal, say "DONE" and nothing else.'
)

# Optional room counting instruction
if self.include_room_count:
    prompt_template += " Also, count the number of rooms you think you have seen."

# Format with the actual goal
prompt = prompt_template.format(goal=natural_language_goal)
```

### Example Full Prompt

With `SceneMemory` and `ObjectNav` task to find a "red chair":

```
You are a robot with four cameras, arranged clockwise as front, right, down, 
and left. Your goal is to find a red chair. You are in a living room. Point 
to a point on the floor to walk towards or an object to approach in service 
of your goal. If you have satisfied your goal, say "DONE" and nothing else.
```

---

## How to Modify the Prompt

### Step 1: Open the File

```bash
vi /weka/prior/mattw/robo_mm_olmo/olmo/data/robot_datasets.py
# or
code /weka/prior/mattw/robo_mm_olmo/olmo/data/robot_datasets.py
```

### Step 2: Navigate to Line 1396

Find the `_construct_prompts` method in the `RobotDataset` class.

### Step 3: Modify the Prompt Template

**Example Modification:**

```python
# Change from:
prompt_template = (
    "You are a robot with four cameras, arranged clockwise as front, "
    "right, down, and left. Your goal is {goal}."
)

# To (example):
prompt_template = (
    "You are an autonomous navigation agent equipped with a multi-camera system. "
    "Your cameras are positioned as follows: front, right, down, and left (clockwise). "
    "Your current objective is to {goal}."
)
```

### Step 4: Modify Instructions (Optional)

```python
# Change from:
prompt_template += (
    " Point to a point on the floor to walk towards "
    "or an object to approach in service of your goal. If you have satisfied "
    'your goal, say "DONE" and nothing else.'
)

# To (example):
prompt_template += (
    " Indicate where you should move next by pointing to either: "
    "(1) a floor location to navigate towards, or "
    "(2) an object to approach. "
    'Once you have accomplished your objective, respond with "DONE" only.'
)
```

### Step 5: Save and Test

After modifying, the cache will need to be rebuilt:

```bash
# Remove old cache
rm -rf $MOLMO_DATA_DIR/robot_datasets/*

# Or force rebuild with overwrite_cache=True in the dataset initialization
```

---

## Setting Up Robot Training

### Training Script Location

```
/weka/prior/mattw/robo_mm_olmo/launch_scripts/train_multitask_model.py
```

### Basic Training Command

```bash
torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py \
    robot_mixture \
    /path/to/checkpoint \
    --robot_memory_setting=SceneMemory \
    --robot_prompt_style=standard \
    --robot_done_behavior=Standard \
    --robot_room_count_behavior=Standard \
    --save_folder=./robot_training \
    --global_batch_size=256 \
    --device_train_batch_size=4 \
    --duration=10000 \
    --wandb.name=robot-training-run \
    --wandb.entity=your-entity \
    --wandb.project=robot-molmo
```

---

## Configuration Options

### Memory Settings (`--robot_memory_setting`)

Controls what context the robot remembers:

| Option | Description | Use Case |
|--------|-------------|----------|
| `NoMemory` | No context about past observations | Baseline, reactive policies |
| `SceneMemory` | Remembers scene descriptions | Standard navigation |
| `SceneAndObjectMemory` | Remembers scenes AND objects seen | Complex multi-room tasks |

**Example:**
```bash
--robot_memory_setting=SceneAndObjectMemory
```

### Prompt Style (`--robot_prompt_style`)

Controls prompt format variant:

| Option | Description | Prompt Content |
|--------|-------------|----------------|
| `standard` | Uses standard prompt template | Camera description + goal + instruction |
| `scene_description` | Uses scene description variant | Alternative formatting with more scene focus |

**Example:**
```bash
--robot_prompt_style=standard
```

### DONE Behavior (`--robot_done_behavior`)

Controls what happens at episode completion:

| Option | Description | Behavior |
|--------|-------------|----------|
| `Standard` | Output "DONE" when goal achieved | Standard completion signal |
| `ObjectPointing` | Point to final object instead of DONE | More explicit final action |

**Example:**
```bash
--robot_done_behavior=ObjectPointing
```

**Note:** `ObjectPointing` is automatically disabled for `ExploreHouse` tasks (doesn't make semantic sense).

### Room Counting (`--robot_room_count_behavior`)

Controls whether to include room counting:

| Option | Description | Effect |
|--------|-------------|--------|
| `Standard` | No room counting | Standard navigation only |
| `RoomCount` | Add room counting task | Adds room count to prompt and actions |

**Example:**
```bash
--robot_room_count_behavior=RoomCount
```

This adds to the prompt:
```
"Also, count the number of rooms you think you have seen."
```

### Task Types

Three main task types available:

| Task Type | Description | Goal |
|-----------|-------------|------|
| `ObjectNav` | Object Navigation | Find and navigate to a specific object |
| `HardObjectNav` | Hard Object Navigation | Find difficult/occluded objects |
| `ExploreHouse` | House Exploration | Explore and map the environment |

Task types are configured in the mixture definition (see Training Mixtures section).

---

## Training Mixtures

### Pre-defined Mixtures

Look in `train_multitask_model.py` for mixture definitions. Common patterns:

```python
# Pure robot training
elif args.mixture == "robot_only":
    robot_tasks = []
    for task in ["ObjectNav", "HardObjectNav", "ExploreHouse"]:
        robot_task_name = _generate_robot_task_name(
            base_task_type=task,
            memory_setting=args.robot_memory_setting,
            prompt_style=args.robot_prompt_style,
            done_behavior=args.robot_done_behavior,
            room_count_behavior=args.robot_room_count_behavior
        )
        robot_tasks.append(robot_task_name)
    
    tasks = [["robot", robot_tasks, 1.0]]  # 100% robot data
```

### Custom Mixture Example

To create your own mixture combining robot and other tasks:

```python
# Add to train_multitask_model.py around line 250+

elif args.mixture == "robot_with_vqa":
    # Robot navigation tasks
    robot_tasks = []
    for task in ["ObjectNav", "ExploreHouse"]:
        robot_task_name = _generate_robot_task_name(
            base_task_type=task,
            memory_setting=args.robot_memory_setting,
            prompt_style=args.robot_prompt_style,
            done_behavior=args.robot_done_behavior
        )
        robot_tasks.append(robot_task_name)
    
    # VQA tasks for general vision understanding
    vqa_tasks = ["coco_2014_vqa_multi", "text_vqa"]
    
    # 70% robot, 30% VQA
    tasks = [
        ["robot", robot_tasks, 0.7],
        ["vqa", vqa_tasks, 0.3]
    ]
    
    # Evaluation tasks
    eval_tasks = robot_tasks + vqa_tasks
```

Then train with:
```bash
torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py robot_with_vqa ...
```

### Task Name Generation Helper

The helper function generates dataset names:

```python
_generate_robot_task_name(
    base_task_type="ObjectNav",           # Task type
    memory_setting="SceneMemory",         # Memory configuration
    prompt_style="standard",              # Prompt variant
    eval_mode=None,                       # Optional eval mode
    is_validation=False,                  # Training vs validation
    done_behavior="Standard",             # DONE behavior
    room_count_behavior="Standard"        # Room counting
)
# Returns: "robot_ObjectNav_SceneMemory"
```

---

## Data Location

### Source Data

VIDA dataset is located at:
```
/weka/prior/datasets/vida_procthor_with_holodeck_assets/2025_07_15/tasks/
├── ObjectNavType/
│   ├── train/
│   └── val/
├── HardObjectNavType/
│   ├── train/
│   └── val/
└── SimpleExploreHouse/
    ├── train/
    └── val/
```

### Cached Data

Processed dataset cache is stored at:
```
$MOLMO_DATA_DIR/robot_datasets/
```

**Cache Files Include:**
- HDF5 shards with preprocessed data
- Index files mapping examples to shards
- Cached prompts and labels

### Managing Cache

**Clear cache to rebuild with new prompts:**
```bash
rm -rf $MOLMO_DATA_DIR/robot_datasets/*
```

**Check cache size:**
```bash
du -sh $MOLMO_DATA_DIR/robot_datasets/
```

---

## Quick Start Examples

### Example 1: Basic Robot Training

```bash
#!/bin/bash
# Basic robot navigation training

export MOLMO_DATA_DIR=/weka/prior/mattw/data
export WANDB_API_KEY=your_wandb_key

torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py \
    robot_mixture \
    /path/to/molmo-7b-checkpoint \
    --robot_memory_setting=SceneMemory \
    --robot_prompt_style=standard \
    --robot_done_behavior=Standard \
    --save_folder=./checkpoints/robot_basic \
    --global_batch_size=256 \
    --device_train_batch_size=4 \
    --duration=10000 \
    --seq_len=2304 \
    --wandb.name=robot-basic-training \
    --wandb.entity=your-team \
    --wandb.project=robot-molmo
```

### Example 2: Advanced Robot Training with Memory

```bash
#!/bin/bash
# Robot training with full memory and room counting

export MOLMO_DATA_DIR=/weka/prior/mattw/data
export WANDB_API_KEY=your_wandb_key

torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py \
    robot_mixture \
    /path/to/molmo-7b-checkpoint \
    --robot_memory_setting=SceneAndObjectMemory \
    --robot_prompt_style=standard \
    --robot_done_behavior=ObjectPointing \
    --robot_room_count_behavior=RoomCount \
    --save_folder=./checkpoints/robot_advanced \
    --global_batch_size=256 \
    --device_train_batch_size=4 \
    --duration=20000 \
    --seq_len=2304 \
    --wandb.name=robot-advanced-training \
    --wandb.entity=your-team \
    --wandb.project=robot-molmo
```

### Example 3: Debug Run (Single GPU)

```bash
#!/bin/bash
# Quick debug run on single GPU

export MOLMO_DATA_DIR=/weka/prior/mattw/data

torchrun --nproc-per-node=1 launch_scripts/train_multitask_model.py \
    debug \
    /path/to/checkpoint \
    --robot_memory_setting=SceneMemory \
    --robot_prompt_style=standard \
    --save_folder=./debug_run \
    --global_batch_size=32 \
    --device_train_batch_size=2 \
    --duration=100 \
    --wandb=null
```

### Example 4: Multi-Node Training

**Node 0 (Master):**
```bash
torchrun \
    --nproc-per-node=8 \
    --nnodes=2 \
    --node-rank=0 \
    --master-addr=node0-ip \
    --master-port=29500 \
    launch_scripts/train_multitask_model.py \
    robot_mixture \
    /path/to/checkpoint \
    --robot_memory_setting=SceneMemory \
    --save_folder=./checkpoints/robot_multinode \
    --global_batch_size=512 \
    --device_train_batch_size=4 \
    --duration=30000
```

**Node 1:**
```bash
torchrun \
    --nproc-per-node=8 \
    --nnodes=2 \
    --node-rank=1 \
    --master-addr=node0-ip \
    --master-port=29500 \
    launch_scripts/train_multitask_model.py \
    robot_mixture \
    /path/to/checkpoint \
    --robot_memory_setting=SceneMemory \
    --save_folder=./checkpoints/robot_multinode \
    --global_batch_size=512 \
    --device_train_batch_size=4 \
    --duration=30000
```

---

## Training Configuration Parameters

### Essential Parameters

```bash
# Checkpoint to start from
/path/to/checkpoint

# Training duration (steps)
--duration=10000

# Batch size (global across all GPUs)
--global_batch_size=256

# Per-device batch size
--device_train_batch_size=4

# Sequence length
--seq_len=2304

# Save location
--save_folder=./checkpoints
```

### Optional Parameters

```bash
# Evaluation batch size
--device_eval_batch_size=4

# Inference batch size  
--device_inf_batch_size=4

# Max inference examples
--max_inf_examples=2048

# Turn off inference during training (faster)
--turn_off_inference

# Include images in eval outputs (for visualization)
--include_image

# Image preprocessing
--max_crops=1
--image_pooling_h=2
--image_pooling_w=2
```

---

## Monitoring Training

### W&B Logging

Monitor training progress in Weights & Biases:

```
https://wandb.ai/your-entity/robot-molmo/runs/your-run-id
```

**Key Metrics to Watch:**
- `train/loss`: Training loss
- `train/throughput`: Tokens/second
- `eval/robot_*_accuracy`: Robot task performance
- `gpu/memory`: GPU memory usage

### Local Checkpoints

Checkpoints saved to:
```
./checkpoints/robot_basic/
├── step0/
├── step1000/
├── step2000/
└── latest/ -> step2000
```

---

## Troubleshooting

### Issue: Cache Not Rebuilding After Prompt Change

**Solution:**
```bash
rm -rf $MOLMO_DATA_DIR/robot_datasets/*
```

### Issue: Out of Memory

**Solutions:**
1. Reduce batch size: `--device_train_batch_size=2`
2. Reduce sequence length: `--seq_len=1792`
3. Use gradient checkpointing (enabled by default)
4. Use fewer GPUs with gradient accumulation

### Issue: Data Not Found

**Check:**
```bash
ls /weka/prior/datasets/vida_procthor_with_holodeck_assets/2025_07_15/tasks/
echo $MOLMO_DATA_DIR
```

### Issue: Slow Training

**Solutions:**
1. Increase data workers: `--data.num_workers=8`
2. Use local SSD for cache
3. Pre-build cache before training

---

## Advanced Topics

### Custom Task Types

To add a new robot task type, modify `robot_datasets.py`:

```python
# Add to TASK_TYPES in RobotDatasetConfig
TASK_TYPES = {
    "ObjectNav": {...},
    "HardObjectNav": {...},
    "ExploreHouse": {...},
    "YourNewTask": {
        "task_type": "YourNewTaskType",
        "vida_subpath": "path/to/your/task/data"
    }
}
```

### Evaluation During Training

Evaluation tasks are defined in the mixture. To add robot validation:

```python
eval_tasks = [
    _generate_robot_task_name(
        base_task_type="ObjectNav",
        memory_setting="SceneMemory",
        prompt_style="standard",
        is_validation=True  # Use validation split
    )
]
```

### Hyperparameter Tuning

Recommended ranges for robot training:

| Parameter | Recommended | Range |
|-----------|-------------|-------|
| Learning Rate | 1e-6 | 5e-7 to 5e-6 |
| Batch Size | 256 | 128 to 512 |
| Duration | 10000 | 5000 to 30000 |
| Sequence Length | 2304 | 1792 to 2304 |

---

## Contact and Support

For questions or issues:
- Check existing GitHub issues
- Create new issue with `[robot]` tag
- Include configuration and error logs

## References

- Main training script: `launch_scripts/train_multitask_model.py`
- Robot dataset: `olmo/data/robot_datasets.py`
- VIDA data utilities: `data/utils/vida_utils.py`
- Visualization utilities: `data/utils/visualization_utils.py`

---

**Last Updated:** 2025-01-07
**Version:** 1.0
