# Robot Datasets

Documentation for robot navigation and manipulation datasets.

## Overview

Robot datasets provide multimodal data for training embodied AI agents:
- Navigation episodes
- Multi-view observations
- Task descriptions
- Action sequences
- Success labels

## Dataset Structure

### RobotDataset

Unified interface for robot navigation data with configurable options.

### Configuration Options

```python
from olmo.data.robot_datasets import RobotDataset, RobotDatasetConfig

# Memory settings
memory_config = RobotDatasetConfig.MEMORY_SETTINGS["SceneAndObjectMemory"]

# Task types
task_config = RobotDatasetConfig.TASK_TYPES["ObjectNav"]

# Create dataset
dataset = RobotDataset(
    task_type="ObjectNav",
    memory_setting="SceneAndObjectMemory",
    split="train"
)
```

## Task Types

### ObjectNav (Object Navigation)

**Goal:** Navigate to find a target object.

**Example:**
```python
{
    "observations": [
        {"rgb": np.ndarray, "depth": np.ndarray},  # Per timestep
        ...
    ],
    "task": "Find the red chair",
    "target_object": "red chair",
    "success": True,
    "trajectory_length": 42
}
```

**Metrics:**
- Success rate
- Path length
- Navigation efficiency

### HardObjectNav

**Goal:** Navigate to difficult-to-find objects.

**Challenges:**
- Occluded objects
- Small objects
- Similar distractors

### ExploreHouse

**Goal:** Explore environment to build mental map.

**Outputs:**
- Coverage percentage
- Exploration trajectory
- Room discovery order

## Memory Settings

### SceneAndObjectMemory

Agent maintains memory of:
- Scene layout
- Object locations
- Previously visited areas

**Best for:** Long-horizon tasks, complex environments

### SceneMemory

Agent maintains only scene layout memory.

**Best for:** Simpler navigation tasks

### NoMemory

No memory, purely reactive policies.

**Best for:** Baseline comparisons

## Data Format

### Episode Structure

```python
{
    "episode_id": str,
    "task_type": str,  # "ObjectNav", etc.
    "target": str,  # Target object or goal
    "observations": List[Dict],  # Per-timestep observations
    "actions": List[int],  # Action sequence
    "success": bool,
    "metadata": {
        "scene": str,
        "difficulty": str,
        "trajectory_length": int,
    }
}
```

### Observations

Each observation contains:

```python
{
    "rgb": np.ndarray,  # [H, W, 3] - RGB image
    "depth": np.ndarray,  # [H, W] - Depth map (optional)
    "egocentric_view": np.ndarray,  # First-person view
    "overhead_view": np.ndarray,  # Bird's-eye view (if available)
    "position": Tuple[float, float, float],  # (x, y, z)
    "rotation": Tuple[float, float, float, float],  # Quaternion
    "timestamp": float,
}
```

## VIDA Dataset

### Description

Large-scale navigation dataset from AI2's VIDA project.

**Environments:**
- ProcTHOR scenes
- Holodeck assets
- Realistic indoor environments

**Statistics:**
- Episodes: ~500K
- Scenes: ~10K unique
- Average episode length: 30-50 steps

### Directory Structure

```
vida_procthor_with_holodeck_assets/
└── 2025_07_15/
    └── tasks/
        ├── ObjectNavType/
        │   ├── train/
        │   └── val/
        ├── HardObjectNavType/
        └── SimpleExploreHouse/
```

### Loading VIDA

```python
from olmo.data.robot_datasets import RobotDataset

dataset = RobotDataset(
    task_type="ObjectNav",
    memory_setting="SceneAndObjectMemory",
    split="train",
    vida_subpath="vida_procthor_with_holodeck_assets/2025_07_15/tasks"
)

episode = dataset[0]
print(f"Task: {episode['task']}")
print(f"Frames: {len(episode['observations'])}")
```

## Training on Robot Data

### Single Task

```bash
torchrun --nproc-per-node=8 launch_scripts/train_robot.py \
    --task_type=ObjectNav \
    --memory_setting=SceneAndObjectMemory \
    --save_folder=./checkpoints
```

### Multi-Task

```python
# Mix different robot tasks
datasets = [
    {"name": "robot_objectnav", "weight": 0.5},
    {"name": "robot_explore", "weight": 0.3},
    {"name": "robot_hard_nav", "weight": 0.2},
]
```

## Evaluation

### Navigation Success Rate

```bash
python launch_scripts/eval_robot.py \
    /path/to/checkpoint \
    --task=ObjectNav \
    --split=validation
```

**Metrics:**
- Success Rate (SR)
- Success weighted by Path Length (SPL)
- Distance to Goal (DTG)

### Qualitative Analysis

Generate trajectory visualizations:

```python
from data.utils.visualization_utils import visualize_trajectory

visualize_trajectory(
    episode=episode,
    output_path="trajectory.mp4"
)
```

## Data Preprocessing

### Frame Selection

```python
from data.utils.vida_utils import select_good_frames

# Select most informative frames
selected_frames = select_good_frames(
    episode,
    max_frames=16,
    method="uniform"  # or "adaptive"
)
```

### Multi-View Composition

```python
from data.utils.visualization_utils import compose_2x2_grid

# Combine multiple views into single image
grid = compose_2x2_grid(
    images=[ego_view, overhead_view, depth, semantic]
)
```

## Objaverse Annotations

### Object Recognition

```python
from data.utils.vida_utils import OBJAVERSE_ANNOTATIONS, SYNSET_TO_BEST_LEMMA

# Get object name from synset
synset = "n02958343"  # car synset
object_name = SYNSET_TO_BEST_LEMMA[synset]
print(f"Object: {object_name}")
```

### Shortest Object Option

```python
from data.utils.vida_utils import get_shortest_objaverse_option

# Get most concise name for object
short_name = get_shortest_objaverse_option(object_id)
```

## Caching

Robot dataset uses LRU caching for efficiency:

```python
# Cache size configurable
RobotDataset._shard_cache_size = 4  # Keep 4 shards in memory
```

### Memory Management

```python
# Monitor memory usage
import psutil

process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.1f} MB")
```

## Best Practices

### Training

1. **Frame Sampling:** Use 8-16 frames per episode
2. **Multi-View:** Include both ego and overhead views
3. **Memory Settings:** Start with SceneAndObjectMemory
4. **Batch Size:** Smaller batches due to sequence length
5. **Mixed Tasks:** Combine navigation and exploration

### Evaluation

1. **Multiple Episodes:** Evaluate on 100+ episodes
2. **Success Threshold:** Use standard SPL metric
3. **Difficulty Levels:** Separate easy/medium/hard
4. **Visualization:** Generate trajectories for analysis

## Troubleshooting

### Slow Loading

**Problem:** Dataset loading is slow.

**Solutions:**
- Increase shard cache size
- Use faster storage (local SSD)
- Reduce frame count per episode
- Prefetch episodes in background

### Memory Issues

**Problem:** Out of memory during training.

**Solutions:**
- Reduce frame count
- Lower image resolution
- Smaller batch size
- Use gradient checkpointing
- Reduce cache size

### Missing Data

**Problem:** "Shard file not found" errors.

**Solutions:**
- Verify VIDA_SOURCE_DIR path
- Check dataset downloaded completely
- Verify permissions on data directory

## Storage Requirements

- **ObjectNav:** ~100GB
- **HardObjectNav:** ~50GB
- **ExploreHouse:** ~50GB
- **Total (all tasks):** ~200GB

## Related Work

- [ALFRED](https://askforalfred.com/)
- [Habitat](https://aihabitat.org/)
- [RoboTHOR](https://ai2thor.allenai.org/)
- [ManipulaTHOR](https://ai2thor.allenai.org/manipulathor/)

## Citation

```bibtex
@article{vida2024,
  title={VIDA: Large-Scale Navigation Dataset},
  author={AI2},
  year={2024}
}
```

## Next Steps

- [Training Guide](../guides/training_guide.md)
- [Video Datasets](video_datasets.md)
- [Custom Datasets](custom_datasets.md)

