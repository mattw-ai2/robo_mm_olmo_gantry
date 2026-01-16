import dataclasses
import functools
import gc
import glob
import hashlib
import json
import logging
import os
import random
from os.path import join
from typing import Dict, List, Optional
import argparse
import sys 
import time

import h5py
import numpy as np
import torchvision.io
from tqdm import tqdm
import psutil
import math
import h5py
import json
import sys
sys.path.append('/weka/prior/mattw/robo_mm_olmo')
from data.utils.vida_utils import OBJAVERSE_ANNOTATIONS, SYNSET_TO_BEST_LEMMA
from data.utils.vida_utils import parse_local_scene_info
import glob

from data.utils.vida_utils import (
    parse_episode_data,
    select_good_frames,
    OBJAVERSE_ANNOTATIONS,
    SYNSET_TO_BEST_LEMMA,
    normalize,
    get_shortest_objaverse_option,
)
from data.utils.visualization_utils import compose_2x2_grid
from olmo.data.dataset import Dataset

# Base directory for robot dataset sources
VIDA_SOURCE_DIR = "/weka/prior/datasets"

# Directory for caching the datasets - should be a subdir of MOLMO_DATA_DIR
MOLMO_DATA_DIR = os.environ.get("MOLMO_DATA_DIR", "/tmp")
CACHE_DIR = os.path.join(MOLMO_DATA_DIR, "robot_datasets", "new_prompts_v1_negative_done_v1.4_CE_Memory")

logger = logging.getLogger(__name__)

# Memory and task configuration for robot datasets
class RobotDatasetConfig:
    # Memory settings
    

    MEMORY_SETTINGS = {
        "SceneAndObjectMemory": {"include_scene": True, "include_objects": True},
        "SceneMemory": {"include_scene": True, "include_objects": False},
        "NoMemory": {"include_scene": False, "include_objects": False},
    }
    
    # Task types
    path1 = 'vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor'
    #path1 = "vida_procthor_with_holodeck_assets/2025_07_15/tasks" # stretch
    path2 = "vida_procthor_with_holodeck_assets/2025_07_15/opentype_tasks" # stretch + open data
    # path = "vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor" # fpin
    
    TASK_TYPES = {
        "ObjectNav": {"task_type": "ObjectNavType", "vida_subpath": f"{path1}", "vida_subpath_source": path1},
        "HardObjectNav": {"task_type": "HardObjectNavType", "vida_subpath": f"{path1}", "vida_subpath_source": path1},
        "ExploreHouse": {"task_type": "SimpleExploreHouse", "vida_subpath": f"{path1}", "vida_subpath_source": path1},
    }

    
    # Eval mode
    EVAL_MODES = {
        "Standard": {"final_frame_only": False, "validation_only": False},
        "DoneEval": {"final_frame_only": True, "validation_only": True},
    }
    
    # DONE frame behavior
    DONE_BEHAVIORS = {
        "Standard": {"done_with_object_points": False},
        "ObjectPointing": {"done_with_object_points": True},
    }
    
    # Room counting behavior
    ROOM_COUNT_BEHAVIORS = {
        "Standard": {"include_room_count": False},
        "RoomCount": {"include_room_count": True},
    }



all_asset_ids = list(OBJAVERSE_ANNOTATIONS.keys())
all_synsets = list(SYNSET_TO_BEST_LEMMA.keys())
# Use the lemma values and clean them
object_list = []
for synset, lemma in SYNSET_TO_BEST_LEMMA.items():
    # Replace underscores with spaces in the lemma
    clean_lemma = lemma.replace('_', ' ')
    object_list.append(clean_lemma)

# Get unique objects and sort - store in global variable
ALL_OBJECTS = sorted(list(set(object_list)))

OBJNAV_TYPES_THOR = [
        "alarm clock",
        "apple",
        "basketball",
        "bed",
        "bowl",
        "chair",
        "garbage can",
        "house plant",
        "laptop",
        "mug",
        "sofa",
        "spray bottle",
        "television",
        "toilet",
        "vase",
    ]


# Add this at the module level, before the RobotDataset class definition
def get_lru_cache_decorator(maxsize):
    """Create an LRU cache decorator with the specified maxsize."""
    return functools.lru_cache(maxsize=maxsize)

class RobotDataset(Dataset):
    """Unified robot dataset class that can be configured at initialization time"""        
    _shard_cache_size = 4  # Hardcoded cache size

    # _actual_load_shard_logic must be defined before _cached_load_shard_logic
    @staticmethod
    def _actual_load_shard_logic(shard_path: str) -> List[Dict]:
        """Load an entire shard file. This is the uncached loading logic."""
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        if not RobotDataset._is_hdf5_file(file_path=shard_path):
            raise ValueError(f"Shard file found but is not hdf5: {shard_path}")
        
        shard_data = []
        with h5py.File(shard_path, 'r') as f:
            # Get all example keys (usually string numbers)
            try:
                example_keys = list(f.keys())
                # Sort keys numerically if possible
                try:
                    example_keys = sorted(example_keys, key=int)
                except:
                    example_keys = sorted(example_keys)

                for key in example_keys:
                    group = f[key]
                    example = {}

                    # Load image data
                    if "image" in group:
                        example["image"] = np.array(group["image"])

                    # Load point cloud data if available
                    if "point_cloud" in group:
                        example["point_cloud"] = np.array(group["point_cloud"])
                    if "point_cloud_mask" in group:
                        example["point_cloud_mask"] = np.array(group["point_cloud_mask"])
                    if "camera_extrinsics" in group:
                        example["camera_extrinsics"] = np.array(group["camera_extrinsics"])

                    # Load text fields
                    for k_bytes in ["prompt_standard", "target_action_standard",
                                    "prompt_scene_description", "target_action_scene_description",
                                    "episode_id", "task_type", "house_id"]:
                        if k_bytes in group:
                            field_data = group[k_bytes][()]
                            if isinstance(field_data, bytes):
                                example[k_bytes] = field_data.decode('utf-8')
                            else:
                                example[k_bytes] = field_data
                        else:
                            example[k_bytes] = ""

                    # Load numeric fields
                    if "frame_idx" in group:
                        example["frame_idx"] = group["frame_idx"][()]
                    else:
                        example["frame_idx"] = 0

                    # Ensure house_id is set even if missing
                    if "house_id" not in example:
                        example["house_id"] = "unknown"

                    shard_data.append(example)
            except Exception as e:
                logger.error(f"Error processing HDF5 file {shard_path}: {e}")
                raise

        logger.info(f"Successfully loaded {len(shard_data)} examples from HDF5 file {os.path.basename(shard_path)}")
        return shard_data

    # Initialize the cached loader at class definition time using the hardcoded size
    _cached_load_shard_logic = staticmethod(
        get_lru_cache_decorator(maxsize=_shard_cache_size)(_actual_load_shard_logic)
    )
    
    def __init__(self, task_type, split, memory_setting="SceneAndObjectMemory", 
                 eval_mode="Standard", sample: Optional[int] = None, non_train_cache_limit: int = 500,
                 debug_mode_sample_limit: Optional[int] = None,
                 load_on_init: bool = True, 
                 prompt_style: str = "standard",
                 done_with_object_points: bool = False,
                 include_room_count: bool = False,
                 include_object_reasoning: bool = False,
                 point_cloud_dir: Optional[str] = None,
                 use_point_cloud: bool = False):
        """
        Initialize a robot dataset.
        
        Args:
            task_type: One of "ObjectNav", "HardObjectNav", "ExploreHouse"
            split: "train" or "validation" (argument for this dataset)
            memory_setting: One of "SceneAndObjectMemory", "SceneMemory", "NoMemory"
            eval_mode: One of "Standard", "DoneEval"
            sample: Optional sample size for in-memory data after loading from files (before caching if applicable).
                     This is distinct from debug_mode_sample_limit.
            non_train_cache_limit: Max examples to save in cache for non-train splits if not in debug mode.
            debug_mode_sample_limit: If set, overrides 'sample' and 'non_train_cache_limit' for debug purposes.
            load_on_init: If False, dataset is initialized but data is not loaded until load() is explicitly called.
            prompt_style: One of "standard", "scene_description". Determines prompt/label format selected by get().
            done_with_object_points: If True, DONE frames will use object points if available 
                                    instead of just returning "DONE"
            include_room_count: If True, adds room counting instruction to prompts and room count to actions
            point_cloud_dir: Directory containing preprocessed point cloud HDF5 files
            use_point_cloud: If True, load and include point cloud data in examples
        """
        super().__init__() 
        
        assert task_type in RobotDatasetConfig.TASK_TYPES, f"Unknown task type: {task_type}"
        assert memory_setting in RobotDatasetConfig.MEMORY_SETTINGS, f"Unknown memory setting: {memory_setting}"
        assert eval_mode in RobotDatasetConfig.EVAL_MODES, f"Unknown eval mode: {eval_mode}"
        assert split in ["train", "validation"], f"Split {split} not supported, must be 'train' or 'validation'"
        assert prompt_style in ["standard", "scene_description"], f"Unknown prompt_style: {prompt_style}"
        
        self.task_type_name = task_type 
        self.eval_mode_name = eval_mode 
        self.is_validation = split == "validation"
        self.prompt_style = "standard"#prompt_style 
        
        self.split = split 
        # Determine effective_split for data sourcing logic: 'DoneEval' implies 'validation'
        self.effective_split_for_data_sourcing = split
        if self.eval_mode_name == "DoneEval" and RobotDatasetConfig.EVAL_MODES[self.eval_mode_name]["validation_only"]:
            self.effective_split_for_data_sourcing = "validation"
        
        # Print split information for debugging
        print(f"[RobotDataset] Creating dataset: task={task_type}, split={split}, eval_mode={eval_mode}, effective_split_for_data_sourcing={self.effective_split_for_data_sourcing}")
        
        self.task_config = RobotDatasetConfig.TASK_TYPES[self.task_type_name]
        self.memory_config = RobotDatasetConfig.MEMORY_SETTINGS[memory_setting]
        self.eval_config = RobotDatasetConfig.EVAL_MODES[self.eval_mode_name]
        
        self.task_type = self.task_config["task_type"] 
        self.VIDA_SUBPATH_SOURCE = self.task_config["vida_subpath_source"]
        self.VIDA_SUBPATH = self.task_config["vida_subpath"]
        self.include_scene = self.memory_config["include_scene"]
        self.include_objects = self.memory_config["include_objects"]
        self.final_frame_only = self.eval_config["final_frame_only"]
        
        self.apply_cache_limit = False
        self.cache_limit_size = 0 
        self.effective_sample_for_load = sample 
        self.done_with_object_points = done_with_object_points
        self.include_room_count = include_room_count
        self.include_object_reasoning = include_object_reasoning
        self.debug_mode = debug_mode_sample_limit is not None
        
        # Point cloud settings
        self.use_point_cloud = use_point_cloud
        self.point_cloud_dir = point_cloud_dir
        if self.use_point_cloud and self.point_cloud_dir is None:
            logger.warning("use_point_cloud is True but point_cloud_dir is not set. Point clouds will not be loaded.")

        if debug_mode_sample_limit is not None:
            logger.info(
                f"DEBUG MODE ACTIVE for {self.task_type_name} (input split {self.split}, effective data split {self.effective_split_for_data_sourcing}, eval {self.eval_mode_name}): "
                f"Applying sample/cache limit of {debug_mode_sample_limit}."
            )
            self.effective_sample_for_load = debug_mode_sample_limit 
            self.apply_cache_limit = True 
            self.cache_limit_size = debug_mode_sample_limit
            if self.cache_limit_size <= 0:
                 logger.warning(f"Debug mode sample limit was <=0 ({self.cache_limit_size}), disabling cache sampling limit.")
                 self.apply_cache_limit = False
        else:
            self.effective_sample_for_load = sample 
            if self.effective_split_for_data_sourcing.lower() != "train":
                if non_train_cache_limit > 0:
                    self.apply_cache_limit = True
                    self.cache_limit_size = non_train_cache_limit
                else:
                    logger.info(f"Non-train cache limit is <=0 ({non_train_cache_limit}) for {self.task_type_name} ({self.effective_split_for_data_sourcing}), no cache sampling limit applied.")
            else: 
                 logger.info(f"Train split for {self.task_type_name} ({self.effective_split_for_data_sourcing}) and not in debug mode. No cache sampling limit applied based on non_train_cache_limit.")

        os.makedirs(CACHE_DIR, exist_ok=True)
        self.data_dir = os.path.join(VIDA_SOURCE_DIR, self.VIDA_SUBPATH_SOURCE)
        self.cache_dir = CACHE_DIR
        
        self.data_index = None  # Will store index for accessing shard data
        self.data = None        # Will store data or reference objects
        
        # Cache is now initialized at class level with a fixed size. No setup needed here.
        logger.info(f"LRU cache for shard loading is active with fixed size: {RobotDataset._shard_cache_size}")
        
        if load_on_init:
            self.data = self.load()
        
    def __len__(self):
        return self.data_index["total_examples"] if self.data_index is not None else 0


    def load(self, overwrite_cache=False, num_workers=None, max_examples_per_shard=10000) -> Optional[List[Dict]]:
        """
        Load examples with efficient caching and sharding.
        
        Args:
            overwrite_cache: Whether to overwrite existing cache files
            num_workers: Number of worker processes for parallel loading
            max_examples_per_shard: Maximum number of examples to store in each shard
        """
        # Store the max shard size
        self.max_examples_per_shard = max_examples_per_shard
        
        # Path for the dataset index that maps to sharded files
        index_path = self._get_index_path()
        
        # Check if index exists and we're not overwriting
        if os.path.exists(index_path) and not overwrite_cache:
            logger.info(f"Loading index from {index_path}")
            with open(index_path, 'r') as f:
                self.data_index = json.load(f)
                
            # Load references to data in shards
            all_data = self._load_minimal_from_shards(self.data_index)
            self.data = all_data
            return self.data
    
        # If no cache or we're overwriting, process from files
        logger.info(f"Building {self.task_type_name} dataset from source files (this may take a while)...")
        data_from_source = self._load_from_files_parallel(num_workers=num_workers, overwrite_shards=overwrite_cache)
        # Apply sampling if needed (consistent with existing code)
        if self.effective_sample_for_load is not None and data_from_source and len(data_from_source) > self.effective_sample_for_load:
            logger.info(f"Sampling {self.effective_sample_for_load} examples from {len(data_from_source)} loaded examples")
            if self.effective_sample_for_load > 0:
                data_from_source = random.sample(data_from_source, self.effective_sample_for_load)
            else:
                data_from_source = []
        
        # Apply cache limit if specified (consistent with existing code)
        data_to_cache = data_from_source
        if self.apply_cache_limit and data_to_cache and len(data_to_cache) > self.cache_limit_size:
            logger.info(f"Limiting cache to {self.cache_limit_size} examples")
            if self.cache_limit_size > 0:
                data_to_cache = random.sample(data_to_cache, self.cache_limit_size)
            else:
                data_to_cache = []
        
        # Create shards and index if we have data to cache
        if data_to_cache:
            # Check if we're dealing with references instead of real data
            if all('_shard_index' in example for example in data_to_cache[:min(5, len(data_to_cache))]):
                logger.info(f"Data contains references only - these are already saved in shards. Skipping additional saving.")
                
                # Just return the data references we already have
                self.data = data_to_cache
                return self.data
            
            # If we get here, we're dealing with real data that needs to be saved
            logger.info(f"Saving {len(data_to_cache)} examples to sharded cache")
            
            # Initialize index data structure
            index_data = {
                "shard_sizes": [],
                "total_examples": len(data_to_cache),
                "shards": []
            }
            
            # Create and save shards
            for shard_id, start_idx in enumerate(range(0, len(data_to_cache), self.max_examples_per_shard)):
                end_idx = min(start_idx + self.max_examples_per_shard, len(data_to_cache))
                shard_examples = data_to_cache[start_idx:end_idx]
                shard_size = len(shard_examples)
                
                shard_path = self._get_shard_path(shard_id)
                logger.info(f"Creating shard {shard_id} at {shard_path} with {shard_size} examples")
                
                try:
                    with h5py.File(shard_path, 'w') as f:
                        for i, example in enumerate(tqdm(shard_examples, desc=f"Caching shard {shard_id}")):
                            group = f.create_group(str(i))
                            group.create_dataset("image", data=example["image"], compression="gzip")
                            # Ensure text data is stored as UTF-8 bytes
                            # Store both standard and scene_description versions
                            group.create_dataset("prompt_standard", data=example["prompt_standard"].encode('utf-8'))
                            group.create_dataset("target_action_standard", data=example["target_action_standard"].encode('utf-8'))
                            group.create_dataset("prompt_scene_description", data=example["prompt_scene_description"].encode('utf-8'))
                            group.create_dataset("target_action_scene_description", data=example["target_action_scene_description"].encode('utf-8'))

                            group.create_dataset("episode_id", data=example["episode_id"].encode('utf-8'))
                            group.create_dataset("frame_idx", data=example["frame_idx"])
                            group.create_dataset("task_type", data=example["task_type"].encode('utf-8'))
                            group.create_dataset("house_id", data=example["house_id"].encode('utf-8'))
                    
                    # Add this shard's info to the index
                    index_data["shard_sizes"].append(shard_size)
                    index_data["shards"].append({"path": shard_path, "shard_id": shard_id})
                    
                except Exception as e:
                    logger.warning(f"Failed to save shard {shard_id}: {e}", exc_info=True)
            
            # Save the index file
            try:
                with open(index_path, 'w') as f:
                    json.dump(index_data, f)
                logger.info(f"Saved index to {index_path}")
            except Exception as e:
                logger.warning(f"Failed to save index: {e}")
        
        # Always use shard references
        self.data_index = index_data
        self.data = self._load_minimal_from_shards(index_data)
        return self.data
    
    def _find_hdf5_files(self, worker_id=None, num_workers=None) -> List[str]:
        """
        Find all HDF5 files for this dataset's task type, using effective_split_for_data_sourcing.
        
        Args:
            worker_id: The ID of the current worker (0-indexed), or None if not using worker-specific assignments
            num_workers: Total number of workers, or None if not using worker-specific assignments
            
        Returns:
            List of HDF5 files assigned to this worker or all files if worker_id is None
        """
        task_dir = join(self.data_dir, self.task_type)
        
        # Use effective_split_for_data_sourcing to determine the sub-directory (train/val)
        split_name_for_path = "val" if self.effective_split_for_data_sourcing == "validation" else self.effective_split_for_data_sourcing
        split_dir = join(task_dir, split_name_for_path)
        
        # Print which directory we're loading from
        print(f"[RobotDataset._find_hdf5_files] Loading data from: {split_dir}")
        print(f"  task_type={self.task_type_name}, split={self.split}, eval_mode={self.eval_mode_name}, effective_split={self.effective_split_for_data_sourcing}")
        
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory {split_dir} not found for {self.task_type} (effective split: {self.effective_split_for_data_sourcing})")
            return []
            
        # Get all files for this split
        glob_pattern = join(split_dir, "**/*.hdf5")
        all_hdf5_files = glob.glob(glob_pattern, recursive=True)
        
        num_found_files = len(all_hdf5_files)

        # If a cache limit is applied, potentially limit the number of files to process
        if self.apply_cache_limit and self.cache_limit_size > 0 and num_found_files > 0:
            # Heuristic: process up to 4x the number of examples we want to cache
            max_files_to_process = 4 * self.cache_limit_size 
            
            if num_found_files > max_files_to_process:
                logger.info(
                    f"Applying file loading limit for {self.task_type} (effective split: {self.effective_split_for_data_sourcing}): "
                    f"Found {num_found_files} files, but will randomly sample "
                    f"{max_files_to_process} files because cache limit is {self.cache_limit_size} examples."
                )
                # Use a consistent seed for reproducibility
                random.seed(42)  
                selected_files = random.sample(all_hdf5_files, max_files_to_process)
                logger.info(f"Selected {len(selected_files)} files out of {num_found_files} for processing for {self.task_type} (effective split: {self.effective_split_for_data_sourcing}).")
            else:
                selected_files = all_hdf5_files
                logger.info(
                    f"For {self.task_type} (effective split: {self.effective_split_for_data_sourcing}), found {num_found_files} files. "
                    f"Cache limit is {self.cache_limit_size} examples, file processing limit is {max_files_to_process}. "
                    f"All found files will be processed."
                )
        else:
            selected_files = all_hdf5_files
            logger.info(f"Found {len(selected_files)} {self.effective_split_for_data_sourcing} files for {self.task_type}.")
        
        # If worker-specific assignments aren't requested, return all selected files
        if worker_id is None or num_workers is None or num_workers <= 1:
            logger.info(f"Found {len(selected_files)} files for {self.task_type}, no worker assignment needed")
            return selected_files
        
        # Create a unique directory for temp files
        temp_dir = os.path.join(CACHE_DIR, "work_assignments")
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Work assignments directory: {temp_dir}")
        
        # Create a unique identifier for this dataset/split combination
        dataset_id = self._get_dataset_id()
        
        # First, check if we have an existing assignment file with matching worker count
        assignment_file = os.path.join(temp_dir, f"{dataset_id}_assignments_{num_workers}.txt")
        logger.info(f"Assignment file path: {assignment_file}")
        
        worker_files = []
        
        # Check if the assignment file exists and matches the current worker count
        if os.path.exists(assignment_file):
            logger.info(f"Found existing assignment file: {assignment_file}")
            try:
                with open(assignment_file, 'r') as f:
                    # The first line should contain the worker count
                    stored_num_workers = int(f.readline().strip())
                    if stored_num_workers != num_workers:
                        logger.warning(f"Worker count mismatch in assignment file. Expected {num_workers}, found {stored_num_workers}. Recreating assignments.")
                        # Continue to the assignment creation below
                    else:
                        # Skip to the worker's section and read their assigned files
                        lines = f.readlines()
                        worker_section_start = -1
                        for i, line in enumerate(lines):
                            if line.strip() == f"WORKER_{worker_id}":
                                worker_section_start = i
                                break
                        
                        if worker_section_start >= 0:
                            # Read files until the next worker section or end of file
                            for i in range(worker_section_start + 1, len(lines)):
                                if lines[i].startswith("WORKER_"):
                                    break
                                file_path = lines[i].strip()
                                if file_path and os.path.exists(file_path):
                                    worker_files.append(file_path)
                        
                            logger.info(f"Worker {worker_id} loaded {len(worker_files)} assigned files from existing assignment file")
                            return worker_files
                        else:
                            logger.warning(f"Worker {worker_id} section not found in assignment file. Recreating assignments.")
                            # Continue to the assignment creation below
            except Exception as e:
                logger.warning(f"Error reading assignment file: {e}. Recreating assignments.")
                # Continue to the assignment creation below
        
        # Create new worker assignments
        logger.info(f"Creating new worker assignments for {dataset_id} with {num_workers} workers in {assignment_file}")
        
        # Shuffle files for better load balancing
        random.shuffle(selected_files)
        
        # Divide files among workers
        worker_assignments = [[] for _ in range(num_workers)]
        for i, file_path in enumerate(selected_files):
            worker_assignments[i % num_workers].append(file_path)
        
        # Log file distribution stats
        file_counts = [len(files) for files in worker_assignments]
        avg_files = sum(file_counts) / len(file_counts) if file_counts else 0
        min_files = min(file_counts) if file_counts else 0
        max_files = max(file_counts) if file_counts else 0
        logger.info(f"File distribution: {len(selected_files)} total files, {avg_files:.1f} files per worker on average (min: {min_files}, max: {max_files})")
        
        # Write assignments to file
        try:
            with open(assignment_file, 'w') as f:
                # Write the worker count as the first line
                f.write(f"{num_workers}\n")
                
                # Write each worker's assignments
                for w_id, files in enumerate(worker_assignments):
                    f.write(f"WORKER_{w_id}\n")
                    for file_path in files:
                        f.write(f"{file_path}\n")
        
            logger.info(f"Successfully wrote worker assignments to {assignment_file}")
        except Exception as e:
            logger.warning(f"Error writing assignment file {assignment_file}: {e}. Proceeding with in-memory assignments.")
        
        # Return the files assigned to this worker
        worker_files = worker_assignments[worker_id]
        logger.info(f"Worker {worker_id} assigned {len(worker_files)} files out of {len(selected_files)} total")
        
        return worker_files
    
    def _load_point_cloud(self, house_id: int, episode_num: str, frame_idx: int) -> Optional[Dict]:
        """
        Load preprocessed point cloud data for a specific frame.
        
        Args:
            house_id: House ID number
            episode_num: Episode number string
            frame_idx: Frame index
            
        Returns:
            Dict with 'point_cloud' and 'point_cloud_mask' or None if not found
        """
        if not self.use_point_cloud or self.point_cloud_dir is None:
            return None
            
        # Construct path to point cloud HDF5 file
        house_id_str = f"{house_id:06d}"
        point_cloud_path = os.path.join(self.point_cloud_dir, house_id_str, "point_clouds.hdf5")
        
        if not os.path.exists(point_cloud_path):
            return None
            
        try:
            with h5py.File(point_cloud_path, 'r') as f:
                # Navigate to the correct frame
                if episode_num not in f:
                    return None
                    
                episode_group = f[episode_num]
                frame_key = str(frame_idx)
                
                if frame_key not in episode_group:
                    # Try to find the closest available frame
                    available_frames = sorted([int(k) for k in episode_group.keys()])
                    if not available_frames:
                        return None
                    # Find the closest frame that is <= frame_idx
                    closest_frame = None
                    for af in available_frames:
                        if af <= frame_idx:
                            closest_frame = af
                    if closest_frame is None:
                        closest_frame = available_frames[0]
                    frame_key = str(closest_frame)
                
                frame_group = episode_group[frame_key]
                
                point_cloud = np.array(frame_group["point_cloud"])
                point_cloud_mask = np.array(frame_group["point_cloud_mask"])
                
                return {
                    "point_cloud": point_cloud,
                    "point_cloud_mask": point_cloud_mask,
                }
                
        except Exception as e:
            logger.debug(f"Error loading point cloud for house {house_id}, episode {episode_num}, frame {frame_idx}: {e}")
            return None

    def _process_file_chunk(self, chunk_files: List[str]) -> List[Dict]:
        """Helper function to process a chunk of HDF5 files for parallel loading."""
        results = []
        # Add this counter for DoneEval logging
        done_frame_stats = {"episodes_processed": 0, "valid_episodes": 0, "episodes_with_good_frames": 0, "total_frames_kept": 0}
        
        # Make tqdm description more specific using effective_split_for_data_sourcing
        progress_bar_desc = f"Worker processing files for {self.task_type} ({self.effective_split_for_data_sourcing})"
        for file_path in tqdm(chunk_files, desc=progress_bar_desc, unit="file"):
            house_id = int(file_path.split('/')[-2])
            # file_results = []  # Track results per file for logging
            try:
                with h5py.File(file_path, "r") as f:
                    episodes = list(f.keys())
                    for episode_num in episodes:
                        done_frame_stats["episodes_processed"] += 1
                        episode_data = f[episode_num]
                        valid_episode, episode_info, _ = parse_episode_data(episode_data)
                        
                        if not valid_episode:
                            continue
                        
                        done_frame_stats["valid_episodes"] += 1
                        
                        # Check if task description is empty - skip if so
                        task_description = episode_info.get("task_description", "")
                        if not task_description or task_description.strip() == "":
                            logger.debug(f"Skipping episode {episode_num} - empty task description")
                            continue
                        
                        # Add logging for DoneEval mode
                        if self.final_frame_only:
                            is_done_frames = episode_info.get("is_done_frame", [])
                            num_done_frames = sum(1 for f in is_done_frames if f)
                            logger.debug(f"Episode {episode_num} has {num_done_frames} DONE frames out of {len(is_done_frames)} total")
                        
                        # Select good frames from this episode
                        max_frames = 10
                        good_frames = select_good_frames(
                            episode_info, 
                            max_to_return=max_frames,
                            final_frame_only=self.final_frame_only
                        )
                        
                        if not good_frames:
                            continue
                        
                        done_frame_stats["episodes_with_good_frames"] += 1
                        if isinstance(good_frames, list):
                            done_frame_stats["total_frames_kept"] += len(good_frames)
                        else:
                            done_frame_stats["total_frames_kept"] += 1
                        
                        # Generate standard prompts and labels
                        standard_prompts = self._construct_prompts(
                            episode_info, good_frames, self.include_scene, self.include_objects
                        )
                        standard_target_actions = self._construct_labels(episode_info, good_frames)

                        # Generate scene description prompts and labels
                        scene_desc_prompts = self._construct_scene_description_prompt(
                            episode_info, good_frames, self.include_scene, self.include_objects
                        )
                        scene_desc_target_actions = self._construct_scene_description_label(
                            episode_info, good_frames, self.include_scene, self.include_objects
                        )
                        
                        # Handle both single and multiple frames
                        current_good_frames = good_frames
                        if not isinstance(current_good_frames, list):
                            current_good_frames = [current_good_frames]
                            standard_prompts = [standard_prompts]
                            standard_target_actions = [standard_target_actions]
                            scene_desc_prompts = [scene_desc_prompts]
                            scene_desc_target_actions = [scene_desc_target_actions]
                        
                        # Process each selected frame
                        for i, frame_idx in enumerate(current_good_frames):
                            frames = {
                                "front": self._get_image(episode_data, "front_rgb", frame_idx),
                                "right": self._get_image(episode_data, "right_rgb", frame_idx),
                                "left": self._get_image(episode_data, "left_rgb", frame_idx),
                                "down": self._get_image(episode_data, "down_rgb", frame_idx),
                            }
                            
                            # If any image is None, skip this frame
                            if any(img is None for img in frames.values()):
                                continue
                            
                            composite_img = compose_2x2_grid(frames)
                            
                            example = {
                                "image": composite_img,
                                "prompt_standard": standard_prompts[i],
                                "target_action_standard": standard_target_actions[i],
                                "prompt_scene_description": scene_desc_prompts[i],
                                "target_action_scene_description": scene_desc_target_actions[i],
                                "episode_id": episode_num,
                                "frame_idx": frame_idx,
                                "include_scene": self.include_scene,
                                "include_objects": self.include_objects,
                                "task_type": f"{self.task_type}",
                                "house_id": house_id,
                            }
                            
                            # Load point cloud data if enabled
                            if self.use_point_cloud:
                                pc_data = self._load_point_cloud(house_id, episode_num, frame_idx)
                                if pc_data is not None:
                                    example["point_cloud"] = pc_data["point_cloud"]
                                    example["point_cloud_mask"] = pc_data["point_cloud_mask"]
                            
                            results.append(example)
                            # Check if this is a DONE frame and create negative/positive augmentation examples
                            is_done_frames = episode_info.get("is_done_frame", [])
                            is_done = frame_idx < len(is_done_frames) and is_done_frames[frame_idx]
                            
                            if is_done:
                                # Get the target object from the original prompt
                                import re
                                natural_language_goal = episode_info.get("task_description", "Unknown goal")
                                task_dict = episode_info.get("task_dict", {})
                                
                                # Extract target object using same logic as line 1457-1477
                                target_object = None
                                
                                # Method 1: Try to extract from synsets (most reliable) - commented out but available
                                # if "synsets" in task_dict and task_dict["synsets"]:
                                #     synset = task_dict["synsets"][0]
                                #     target_object = synset.split('.')[0]  # 'houseplant' from 'houseplant.n.01'
                                
                                # Method 2: Parse natural language description with regex
                                if target_object is None:
                                    match = re.search(r'(?:find|locate|navigate to|go to|search for)\s+(?:a|an|the)\s+(.+)', natural_language_goal, re.IGNORECASE)
                                    if match:
                                        target_object = match.group(1).strip()
                                    else:
                                        target_object = natural_language_goal
                                
                                # Use extracted target_object instead of the empty natural_language_goal
                                target_object_name = target_object
                                
                                # Augmentation 1: Negative object prompt → "There are none." label
                                negative_object = random.choice(OBJNAV_TYPES_THOR)
                                prompt_negative = (
                                    f"Point to {negative_object}\n"
                                    f"Please say 'There are none.' if it is not in the image. "
                                    f"If you point to the {negative_object}, and are close in proximity to the {negative_object}, "
                                    f"say 'DONE' while pointing to the {negative_object}."
    )
                                
                                # Augmentation 1: Negative object prompt → "There are none." label
                                negative_object = random.choice(OBJNAV_TYPES_THOR)
                                prompt_negative = (
                                    f"Point to {negative_object}\n"
                                    f"Please say 'There are none.' if it is not in the image. "
                                    f"If you point to the {negative_object}, and are close in proximity to the {negative_object}, "
                                    f"say 'DONE' while pointing to the {negative_object}."
                                )
                                
                                example_negative = {
                                    "image": composite_img,
                                    "prompt_standard": prompt_negative,
                                    "target_action_standard": "There are none.",
                                    "prompt_scene_description": scene_desc_prompts[i],
                                    "target_action_scene_description": scene_desc_target_actions[i],
                                    "episode_id": episode_num,
                                    "frame_idx": frame_idx,
                                    "include_scene": self.include_scene,
                                    "include_objects": self.include_objects,
                                    "task_type": f"{self.task_type}",
                                    "house_id": house_id,
                                }
                                results.append(example_negative)
                                
                                # Augmentation 2: Positive object prompt → DONE label (duplicate to balance)
                                prompt_positive = (
                                    f"Point to {target_object_name}\n"
                                    f"Please say 'There are none.' if it is not in the image. "
                                    f"If you point to the {target_object_name}, and are close in proximity to the {target_object_name}, "
                                    f"say 'DONE' while pointing to the {target_object_name}."
                                )
                                
                                example_positive = {
                                    "image": composite_img,
                                    "prompt_standard": prompt_positive,
                                    "target_action_standard": standard_target_actions[i],  # DONE with coordinates
                                    "prompt_scene_description": scene_desc_prompts[i],
                                    "target_action_scene_description": scene_desc_target_actions[i],
                                    "episode_id": episode_num,
                                    "frame_idx": frame_idx,
                                    "include_scene": self.include_scene,
                                    "include_objects": self.include_objects,
                                    "task_type": f"{self.task_type}",
                                    "house_id": house_id,
                                }
                                results.append(example_positive)
                            
            except Exception as e:
                logger.error(f"Error processing file {file_path} for {self.task_type} ({self.effective_split_for_data_sourcing}): {e}")
        
        # Log stats at the end of chunk processing for DoneEval mode
        if self.final_frame_only:
            logger.info(f"DoneEval stats for chunk: Processed {done_frame_stats['episodes_processed']} episodes, "
                       f"{done_frame_stats['valid_episodes']} valid episodes, "
                       f"{done_frame_stats['episodes_with_good_frames']} episodes with good frames, "
                       f"{done_frame_stats['total_frames_kept']} total frames kept, "
                       f"{len(results)} final examples")
        
        return results
        
    def _process_chunk_with_shard(self, worker_id, chunk_files, chunk_idx, num_chunks, overwrite_shards=False):
        """Process a chunk of files and save shard"""
        # Create a unique directory for shards
        temp_dir = os.path.join(CACHE_DIR, "shards")
        os.makedirs(temp_dir, exist_ok=True)
        
        dataset_id = self._get_dataset_id()
        
        # Create a hash of the chunk files for identification
        # Sort files to ensure hash stability even if ordering changes
        sorted_files = sorted(chunk_files)
        file_list_str = "\n".join(sorted_files)
        chunk_hash = hashlib.md5(file_list_str.encode('utf-8')).hexdigest()[:3]
        
        # shard file path - changed from .pkl to .h5
        shard_file = os.path.join(
            temp_dir, 
            f"{dataset_id}_worker{worker_id}_chunk{chunk_idx}of{num_chunks}_{chunk_hash}.h5"
        )

        os.makedirs(os.path.dirname(shard_file), exist_ok=True)

        logger.info(
            f"Worker {worker_id}: Processing chunk {chunk_idx}/{num_chunks} "
            f"({len(chunk_files)} files), shard hash: {chunk_hash}"
        )
        
        # Create a debug file listing the chunk contents
        debug_file = os.path.join(
            temp_dir, 
            f"{dataset_id}_worker{worker_id}_chunk{chunk_idx}of{num_chunks}_{chunk_hash}.txt"
        )
        try:
            with open(debug_file, 'w') as f:
                f.write(f"Worker {worker_id}, Chunk {chunk_idx}/{num_chunks}, Hash {chunk_hash}\n")
                f.write(f"File list ({len(chunk_files)} files):\n")
                for file_path in sorted_files:
                    f.write(f"{file_path}\n")
        except Exception as e:
            logger.warning(f"Worker {worker_id}: Error writing debug file: {e}")
        
        # Check if shard exists and should be used
        shard_exists = os.path.exists(shard_file)
        logger.info(f"Worker {worker_id}: shard file {os.path.basename(shard_file)} exists? {shard_exists}")
        
        if shard_exists and not overwrite_shards:
            logger.info(f"Worker {worker_id}: LOADING shard for chunk {chunk_idx}/{num_chunks} with hash {chunk_hash}")
            try:
                start_time = time.time()
                # Changed: Load from HDF5 instead of pickle
                results = []
                with h5py.File(shard_file, 'r') as f:
                    num_examples = len(f.keys())
                    for ex_id in range(num_examples):
                        group = f[str(ex_id)]
                        example = {}
                        # Load image data
                        if "image" in group:
                            example["image"] = np.array(group["image"])
                        
                        # Load text fields
                        for k_bytes in ["prompt_standard", "target_action_standard",
                                        "prompt_scene_description", "target_action_scene_description",
                                        "episode_id", "task_type", "house_id"]:
                            if k_bytes in group:
                                field_data = group[k_bytes][()]
                                if isinstance(field_data, bytes):
                                    example[k_bytes] = field_data.decode('utf-8')
                                else:
                                    example[k_bytes] = field_data
                            else:
                                example[k_bytes] = ""
                        
                        # Load numeric fields
                        if "frame_idx" in group:
                            example["frame_idx"] = group["frame_idx"][()]
                        else:
                            example["frame_idx"] = 0
                        
                        results.append(example)
                        
                load_time = time.time() - start_time
                logger.info(
                    f"Worker {worker_id}: SUCCESSFULLY LOADED shard for chunk {chunk_idx}/{num_chunks} "
                    f"with {len(results)} examples in {load_time:.2f} seconds"
                )
                return results
            except Exception as e:
                logger.warning(f"Worker {worker_id}: ERROR loading shard {shard_file}: {e}. Reprocessing chunk.")
        elif shard_exists and overwrite_shards:
            logger.info(f"Worker {worker_id}: shard exists but overwrite_shards=True. Skipping shard load.")
        else:
            logger.info(f"Worker {worker_id}: No matching shard found with hash {chunk_hash}.")
        
        # Process the chunk
        start_time = time.time()
        chunk_results = self._process_file_chunk(chunk_files)
        process_time = time.time() - start_time
        
        # Log processing statistics
        examples_per_second = len(chunk_results) / process_time if process_time > 0 else 0
        files_per_second = len(chunk_files) / process_time if process_time > 0 else 0
        examples_per_file = len(chunk_results) / len(chunk_files) if chunk_files else 0
        
        logger.info(
            f"Worker {worker_id}: Processed chunk {chunk_idx}/{num_chunks} in {process_time:.2f} seconds. "
            f"Rate: {files_per_second:.2f} files/sec, {examples_per_second:.2f} examples/sec, "
            f"{examples_per_file:.2f} examples/file on average"
        )
        
        # Save shard - Changed: Save as HDF5 instead of pickle
        try:
            start_time = time.time()
            with h5py.File(shard_file, 'w') as f:
                for i, example in enumerate(tqdm(chunk_results, desc=f"Worker {worker_id}: Saving shard {chunk_idx}/{num_chunks}")):
                    group = f.create_group(str(i))
                    # Save image data with compression
                    if "image" in example and example["image"] is not None:
                        group.create_dataset("image", data=example["image"], compression="gzip")
                    
                    # Save text fields as UTF-8 bytes
                    fields_to_save = {
                        "prompt_standard": example.get("prompt_standard"),
                        "target_action_standard": example.get("target_action_standard"),
                        "prompt_scene_description": example.get("prompt_scene_description"),
                        "target_action_scene_description": example.get("target_action_scene_description"),
                        "episode_id": example.get("episode_id"),
                        "task_type": example.get("task_type"),
                        "house_id": example.get("house_id")
                    }

                    for k, v_text in fields_to_save.items():
                        if v_text is not None:
                            if isinstance(v_text, str):
                                group.create_dataset(k, data=v_text.encode('utf-8'))
                            else: # Should be string, but convert just in case
                                group.create_dataset(k, data=str(v_text).encode('utf-8'))
                        else:
                             # Store empty string or a specific placeholder if field is missing
                            default_empty = "unknown" if k == "house_id" else ""
                            group.create_dataset(k, data=default_empty.encode('utf-8'))
                    
                    # Save numeric data
                    if "frame_idx" in example:
                        group.create_dataset("frame_idx", data=example["frame_idx"])
                    else:
                        group.create_dataset("frame_idx", data=0)
                        
            save_time = time.time() - start_time
            
            # Get file size
            shard_size_mb = os.path.getsize(shard_file) / (1024 * 1024)
            
            logger.info(
                f"Worker {worker_id}: Saved shard for chunk {chunk_idx}/{num_chunks} "
                f"with {len(chunk_results)} examples, size: {shard_size_mb:.2f} MB, "
                f"time: {save_time:.2f} seconds"
            )
        except Exception as e:
            logger.warning(f"Worker {worker_id}: Error saving shard {shard_file}: {e}")
        
        # Add this after processing the chunk and saving the shard
        # Force garbage collection to release memory
        import gc
        gc.collect()
        
        return chunk_results

    def _worker_process_chunks(self, worker_id, num_workers, overwrite_shards=False):
        """Worker process function that handles a set of files divided into chunks"""
        # Get files assigned to this worker
        worker_files = self._find_hdf5_files(worker_id=worker_id, num_workers=num_workers)
        
        if not worker_files:
            logger.warning(f"Worker {worker_id}: No files assigned. Returning empty result.")
            return {'total_examples': 0, 'shards': []}

        # Divide into 15 chunks
        num_chunks = 15
        chunk_size = max(1, len(worker_files) // num_chunks)
        chunks = [worker_files[i:i+chunk_size] for i in range(0, len(worker_files), chunk_size)]
        
        # Log chunk distribution
        chunk_sizes = [len(chunk) for chunk in chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
        max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
        
        logger.info(
            f"Worker {worker_id}: Split {len(worker_files)} files into {len(chunks)} chunks. "
            f"Avg {avg_chunk_size:.1f} files per chunk (min: {min_chunk_size}, max: {max_chunk_size})"
        )
        
        # Process each chunk sequentially, only storing shard metadata
        shard_metadata = []
        total_examples = 0
        
        # Process each chunk sequentially, only storing shard metadata
        shard_metadata = []
        total_examples = 0

        # Add tqdm for chunks
        for chunk_idx, chunk_files in enumerate(tqdm(chunks, desc=f"Worker {worker_id} chunks", unit="chunk")):
            # Process the chunk and save to shard
            chunk_results = self._process_chunk_with_shard(
                worker_id, chunk_files, chunk_idx+1, len(chunks), overwrite_shards
            )
            # Only store metadata about this chunk
            num_examples_in_chunk = len(chunk_results)
            total_examples += num_examples_in_chunk
            
            # Create a unique directory for shards
            temp_dir = os.path.join(CACHE_DIR, "shards")
            os.makedirs(temp_dir, exist_ok=True)
            
            dataset_id = self._get_dataset_id()

            # Create a hash of the chunk files for identification
            sorted_files = sorted(chunk_files)
            file_list_str = "\n".join(sorted_files)
            chunk_hash = hashlib.md5(file_list_str.encode('utf-8')).hexdigest()[:3]
            
            # shard file path - changed from .pkl to .h5
            shard_file = os.path.join(
                temp_dir, 
                f"{dataset_id}_worker{worker_id}_chunk{chunk_idx+1}of{len(chunks)}_{chunk_hash}.h5"
            )
            
            # Add metadata for this shard
            shard_metadata.append({
                'path': shard_file,
                'num_examples': num_examples_in_chunk,
                'chunk_hash': chunk_hash,
                'chunk_idx': chunk_idx+1,
                'num_chunks': len(chunks)
            })
            
            # Explicitly release the memory for chunk_results
            del chunk_results
            
            # Force garbage collection
            gc.collect()
            
            # Log progress and memory usage
            logger.info(
                f"Worker {worker_id}: Processed {chunk_idx+1}/{len(chunks)} chunks, "
                f"{total_examples} examples so far, {num_examples_in_chunk} from current chunk"
            )
            
            # Optional: log memory usage if psutil is available
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                logger.info(f"Worker {worker_id}: Memory usage after chunk {chunk_idx+1}: {memory_mb:.1f} MB")
            except ImportError:
                pass  # psutil not installed, skip memory logging
        
        logger.info(f"Worker {worker_id}: Completed all {len(chunks)} chunks, total {total_examples} examples")
        
        # Return metadata instead of actual examples
        return {
            'total_examples': total_examples,
            'shards': shard_metadata
        }

    def _load_from_files_parallel(self, num_workers=None, overwrite_shards=False) -> List[Dict]:
        """Load examples using multiple processes for speed, with chunking for memory efficiency"""
        import multiprocessing as mp
        
        logger.info(f"PARALLEL_LOAD: Starting for {self.task_type} (effective_split: {self.effective_split_for_data_sourcing}), overwrite_shards={overwrite_shards}")
        num_workers = num_workers or max(1, mp.cpu_count() - 1)

        # Single worker case is simpler (no multiprocessing)
        if num_workers <= 1:
            worker_id = 0
            logger.info(f"PARALLEL_LOAD: Processing with a single worker (no multiprocessing.Pool)")
            # For single worker, process directly to shards
            metadata = self._worker_process_chunks(worker_id, 1, overwrite_shards)
            
            # Use shards directly as shards
            self._create_index_from_shards(metadata)

            return self._load_minimal_from_shards(metadata)
        
        # Multiple worker case using multiprocessing
        else:
            logger.info(f"PARALLEL_LOAD: Using {num_workers} workers with chunking")
            
            # Execute worker processes
            try:
                with mp.Pool(num_workers) as pool:
                    logger.info(f"PARALLEL_LOAD: multiprocessing.Pool CREATED with {num_workers} workers.")
                    # Map worker function to worker IDs
                    worker_ids = list(range(num_workers))
                    # Create a partial function that includes the parameters
                    from functools import partial
                    worker_func = partial(self._worker_process_chunks, num_workers=num_workers, overwrite_shards=overwrite_shards)
                    
                    # Add overall progress bar with ETA
                    all_worker_metadata = []
                    print(f"\n{'='*80}", flush=True)
                    print(f"OVERALL PROGRESS: {self.task_type} ({self.effective_split_for_data_sourcing}) - {num_workers} workers", flush=True)
                    print(f"{'='*80}\n", flush=True)
                    
                    with tqdm(total=num_workers, 
                             desc=f"🚀 {self.task_type} workers", 
                             unit="worker",
                             position=0,
                             leave=True,
                             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                        
                        for worker_meta in pool.imap_unordered(worker_func, worker_ids):
                            all_worker_metadata.append(worker_meta)
                            total_examples = sum(m['total_examples'] for m in all_worker_metadata)
                            avg_examples = total_examples // len(all_worker_metadata) if all_worker_metadata else 0
                            pbar.set_postfix_str(f"examples={total_examples:,}, avg={avg_examples:,}/worker")
                            pbar.update(1)
                    
                    logger.info(f"PARALLEL_LOAD: All workers completed successfully.")
                
                # After the worker.map call, add logging for DoneEval
                if self.final_frame_only:
                    logger.info(f"DoneEval worker results breakdown: {[meta['total_examples'] for meta in all_worker_metadata]}")
                
                # Combine the metadata
                combined_metadata = {
                    'total_examples': sum(meta['total_examples'] for meta in all_worker_metadata),
                    'shards': []
                }
                
                for worker_meta in all_worker_metadata:
                    combined_metadata['shards'].extend(worker_meta['shards'])
                
                if self.final_frame_only:
                    logger.info(f"DoneEval combined metadata shows {combined_metadata['total_examples']} examples from all workers")
                
                print(f"\n{'='*80}", flush=True)
                print(f"✅ COMPLETED: {self.task_type} - Total: {combined_metadata['total_examples']:,} examples", flush=True)
                print(f"{'='*80}\n", flush=True)
                
                # Use shards directly as shards
                self._create_index_from_shards(combined_metadata)

                return self._load_minimal_from_shards(combined_metadata)
                
            except Exception as e:
                logger.error(f"PARALLEL_LOAD: Exception during multiprocessing: {e}", exc_info=True)
                return [] 
    
    def _get_cache_file_path_base(self) -> str:
        """Helper function to get the base path string for cache files (shards/index)."""
        memory_setting_str = ""
        if not self.include_scene and not self.include_objects:
            memory_setting_str = "_NoMemory"
        elif self.include_scene and not self.include_objects:
            memory_setting_str = "_SceneMemory"
        elif self.include_scene and self.include_objects:
            memory_setting_str = "_SceneAndObjectMemory"
        else:
            logger.warning(
                f"Unexpected combination of include_scene ({self.include_scene}) "
                f"and include_objects ({self.include_objects}). "
                f"Defaulting memory_setting_str for path."
            )
            memory_setting_str = "_SceneAndObjectMemory"

        eval_mode_suffix = ""
        if self.eval_mode_name == "DoneEval":
            eval_mode_suffix = "_DoneEval"
        elif self.eval_mode_name == "Standard":
            eval_mode_suffix = "_StandardEval"
        else:
            eval_mode_suffix = f"_{self.eval_mode_name}Eval"

        # Include done behavior in cache path
        done_behavior_suffix = ""
        if self.done_with_object_points:
            done_behavior_suffix = "_ObjectPointing"
        
        # Include room count behavior in cache path
        room_count_suffix = ""
        if self.include_room_count:
            room_count_suffix = "_RoomCount"
        
        cache_base = f"{self.VIDA_SUBPATH}_{self.task_type}{memory_setting_str}{eval_mode_suffix}{done_behavior_suffix}{room_count_suffix}_{self.effective_split_for_data_sourcing}"
        if self.debug_mode:
            print(f"DEBUG _get_cache_file_path_base(): base_name={cache_base}, cache_dir={self.cache_dir}", flush=True)
        return cache_base


    def _get_shard_path(self, shard_id):
        """Get path for a specific shard file."""
        base_name = self._get_cache_file_path_base()
        shard_path = os.path.join(
            self.cache_dir,
            f"{base_name}_shard{shard_id}_cache.h5"
        )
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)
        if self.debug_mode:
            print(f"DEBUG: Final shard path for shard_id {shard_id}: {shard_path}", flush=True)
        return shard_path

    # Modify _create_index_from_shards to _create_index_from_shards and add single shard consolidation
    def _create_index_from_shards(self, metadata):
        """Create a dataset index that treats shard files as shards"""
        total_examples = metadata['total_examples']
        shard_infos = sorted(metadata['shards'], key=lambda x: (x.get('chunk_idx', 0), x.get('path', '')))
        
        # Apply sampling if needed
        max_examples = total_examples
        if self.effective_sample_for_load is not None and self.effective_sample_for_load > 0 and total_examples > self.effective_sample_for_load:
            max_examples = self.effective_sample_for_load
            logger.info(f"Will sample {max_examples} examples from {total_examples} total")
        
        # Apply cache limit if specified
        if self.apply_cache_limit and self.cache_limit_size > 0 and total_examples > self.cache_limit_size:
            max_examples = min(max_examples, self.cache_limit_size)
            logger.info(f"Applying cache limit: will keep {max_examples} examples")
        
        # Check if we should consolidate shards for validation or smaller datasets
        is_small_dataset = (self.effective_split_for_data_sourcing == "validation" or 
                            self.eval_mode_name == "DoneEval" or
                            max_examples < self.max_examples_per_shard)
                            
        should_consolidate = is_small_dataset and len(shard_infos) > 1
        
        if should_consolidate:
            logger.info(f"Detected small dataset ({max_examples} examples) with {len(shard_infos)} shards. Will consolidate into a single shard.")
            
            # Collect all examples up to max_examples
            all_examples = []
            examples_collected = 0
            
            for shard_info in shard_infos:
                if examples_collected >= max_examples:
                    break
                    
                shard_path = shard_info['path']
                try:
                    # Load from HDF5
                    with h5py.File(shard_path, 'r') as f:
                        # Count examples
                        example_keys = list(f.keys())
                        examples_to_take = min(len(example_keys), max_examples - examples_collected)
                        
                        # Sort keys numerically if possible
                        try:
                            example_keys = sorted(example_keys, key=int)
                        except:
                            example_keys = sorted(example_keys)
                            
                        # Load only what we need
                        for i in range(examples_to_take):
                            key = example_keys[i]
                            group = f[key]
                            example = {}
                            
                            # Load image data
                            if "image" in group:
                                example["image"] = np.array(group["image"])
                            
                            # Load text fields
                            for k_bytes in ["prompt_standard", "target_action_standard",
                                            "prompt_scene_description", "target_action_scene_description",
                                            "episode_id", "task_type", "house_id"]:
                                if k_bytes in group:
                                    field_data = group[k_bytes][()]
                                    if isinstance(field_data, bytes):
                                        example[k_bytes] = field_data.decode('utf-8')
                                    else:
                                        example[k_bytes] = field_data
                                else:
                                    example[k_bytes] = ""
                            
                            # Load numeric fields
                            if "frame_idx" in group:
                                example["frame_idx"] = group["frame_idx"][()]
                            else:
                                example["frame_idx"] = 0
                            
                            all_examples.append(example)
                        
                        examples_collected += examples_to_take
                
                    # Force garbage collection after each shard
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Error loading shard {shard_path}: {e}")
            
            # Create a single consolidated shard
            consolidated_shard_path = self._get_shard_path(0)
            logger.info(f"Creating consolidated shard at {consolidated_shard_path} with {len(all_examples)} examples")
            
            try:
                # Save the consolidated shard
                with h5py.File(consolidated_shard_path, 'w') as f:
                    for i, example in enumerate(tqdm(all_examples, desc=f"Creating consolidated shard")):
                        group = f.create_group(str(i))
                        # Check if "image" exists in the example
                        if "image" not in example:
                            logger.warning(f"Example {i} is missing 'image' field. Skipping.")
                            continue
                        
                        # Save the data fields
                        group.create_dataset("image", data=example["image"], compression="gzip")
                        
                        # Ensure text data is stored as UTF-8 bytes
                        for field in ["prompt_standard", "target_action_standard",
                                      "prompt_scene_description", "target_action_scene_description",
                                      "episode_id", "task_type", "house_id"]:
                            if field in example and example[field] is not None:
                                group.create_dataset(field, data=example[field].encode('utf-8'))
                            else:
                                # If the field is missing, store an empty string
                                group.create_dataset(field, data="".encode('utf-8'))
                        
                        # Save numeric data
                        if "frame_idx" in example:
                            group.create_dataset("frame_idx", data=example["frame_idx"])
                        else:
                            group.create_dataset("frame_idx", data=0)
                
                # Create index with single shard
                index_data = {
                    "shard_sizes": [len(all_examples)],
                    "total_examples": len(all_examples),
                    "shards": [{"path": consolidated_shard_path, "shard_id": 0}],  # Add shards key for streaming mode
                    "created_timestamp": time.time()
                }
                
                # Save the index
                index_path = self._get_index_path()
                with open(index_path, 'w') as f:
                    json.dump(index_data, f)
                
                logger.info(f"Created consolidated index with 1 shard, total {len(all_examples)} examples")
                
                # Store index in instance
                self.data_index = index_data
                return index_data
                
            except Exception as e:
                logger.warning(f"Failed to create consolidated shard: {e}", exc_info=True)
                # Fall back to non-consolidated approach if consolidation fails
        
        # If not consolidating or consolidation failed, proceed with original approach
        shard_sizes = []
        examples_accounted_for = 0
        
        for shard_info in shard_infos:
            num_examples = shard_info.get('num_examples', 0)
            
            # If we need to limit examples, apply limit to each shard proportionally
            if examples_accounted_for + num_examples > max_examples:
                effective_examples = max_examples - examples_accounted_for
                if effective_examples <= 0:
                    break
                shard_sizes.append(effective_examples)
                examples_accounted_for += effective_examples
                break
            else:
                shard_sizes.append(num_examples)
                examples_accounted_for += num_examples
        
        # Create and save the index
        index_data = {
            "shard_sizes": shard_sizes,
            "total_examples": examples_accounted_for,
            "shards": shard_infos[:len(shard_sizes)],  # Only keep used shards
            "created_timestamp": time.time()
        }
        
        # Save the index
        index_path = self._get_index_path()
        with open(index_path, 'w') as f:
            json.dump(index_data, f)
        
        logger.info(f"Created index with {len(shard_sizes)} shards, total {examples_accounted_for} examples")
        
        # Store index in instance
        self.data_index = index_data
        return index_data

    def _load_minimal_from_shards(self, metadata):
        """
        Load minimal metadata for non-streaming mode using shard files directly.
        This avoids loading all the actual data into memory.
        """
        # Create the index if not already done
        if self.data_index is None:
            self._create_index_from_shards(metadata)
        
        # Create a list of reference objects without loading actual data
        data_references = []
        total_examples = self.data_index.get("total_examples", 0)
        
        for idx in range(total_examples):
            data_references.append({
                "_shard_index": idx,  # Store the index for lazy loading
            })
        
        logger.info(f"Created {len(data_references)} data references for lazy loading from shards")
        return data_references


    # This method uses the instance-specific LRU cache.
    def _load_shard(self, shard_path: str) -> List[Dict]:
        """Load and cache an entire shard file using the class-level cache."""
        return self._cached_load_shard_logic(shard_path)

    
    def get(self, idx, rng=None):
        """Simplified get method with direct access pattern"""
        if self.data_index is None:
            self.data_index = self.load_index()
            
        # Find shard location
        shard_idx, example_id = self._get_shard_and_example_id(idx)
        shard_info = self.data_index["shards"][shard_idx]
        shard_path = shard_info["path"]
        
        # Load and cache the shard, get the example
        try:
            shard_data = self._load_shard(shard_path)
            item = shard_data[example_id]
        except Exception as e:
            logger.error(f"Error retrieving example {idx}: {e}")
            # Return placeholder data
            item = self._create_error_placeholder(idx, str(e))
        
        # Select prompt and target_action based on self.prompt_style
        prompt_to_use = ""
        target_action_to_use = ""

        if self.prompt_style == "scene_description":
            prompt_to_use = item.get("prompt_scene_description", "Error: scene_description prompt missing")
            target_action_to_use = item.get("target_action_scene_description", "Error: scene_description label missing")
        else: # Standard
            prompt_to_use = item.get("prompt_standard", "Error: standard prompt missing")
            target_action_to_use = item.get("target_action_standard", "Error: standard label missing")

        is_done_frame = "DONE" in target_action_to_use
        
        result = {
            "image": item.get("image"),
            "prompt": prompt_to_use,
            "text": target_action_to_use,
            "style": "pointing",
            "metadata": {
                "image": item.get("image"), # item.get("image") might be large, consider if needed here
                "prompt": prompt_to_use, # Redundant with outer prompt, but often useful in metadata
                "target_action": target_action_to_use, # Redundant, same reason
                "episode_id": item.get("episode_id", "unknown"),
                "house_id": item.get("house_id", "unknown"),
                "frame_idx": item.get("frame_idx", -1),
                "task_type": item.get("task_type", self.task_type),
                "include_scene": self.include_scene,
                "include_objects": self.include_objects,
                "is_done_frame": is_done_frame,
                "target_box": None,
            },
        }
        
        # Add point cloud data if available
        if "point_cloud" in item and item["point_cloud"] is not None:
            result["point_cloud"] = item["point_cloud"]
        if "point_cloud_mask" in item and item["point_cloud_mask"] is not None:
            result["point_cloud_mask"] = item["point_cloud_mask"]
        if "camera_extrinsics" in item and item["camera_extrinsics"] is not None:
            result["camera_extrinsics"] = item["camera_extrinsics"]
            
        return result

    @staticmethod
    def _is_hdf5_file(file_path):
        """Detect if a file is in HDF5 format."""
        if not os.path.exists(file_path):
            return False
            
        # Check file extension first
        if file_path.endswith(".h5") or file_path.endswith(".hdf5"):
            return True
            
        # Try to open as HDF5
        try:
            with h5py.File(file_path, 'r') as _:
                return True
        except:
            return False

    def _get_image(self, episode_data, camera_key, frame_idx):
        """Extract and process image from video files associated with the episode."""
        camera_name = camera_key.replace("_rgb", "")
        
        # Get the HDF5 file path from the episode data
        hdf5_path = episode_data.file.filename
        base_dir = os.path.dirname(hdf5_path)
        episode_num = os.path.basename(episode_data.name)
        
        # Try both raw and warped camera paths
        for prefix in ["raw", "warped"]:
            video_path = os.path.join(
                base_dir, f"{prefix}_{camera_name}_camera__{episode_num}.mp4"
            )
            if os.path.exists(video_path):
                try:
                    video = torchvision.io.read_video(video_path, pts_unit="sec")[0]
                    if frame_idx < len(video):
                        return video[frame_idx].numpy()
                except Exception as e:
                    logger.warning(f"Error reading video {video_path}: {e}")
        
        return None
    
    def _construct_prompts(
        self, result, frame_idx=None, include_scene=True, include_objects=False
    ):
        """Constructs a prompt for a robot based on the episode data."""
        if frame_idx is None or not result or not result.get("valid", False):
            raise ValueError("you have to provide episode information and a valid frame index")
        
        # Handle both single frame and multiple frames
        single_frame = not isinstance(frame_idx, list)
        frames_to_process = [frame_idx] if single_frame else frame_idx
        
        prompts = []
        for idx in frames_to_process:
            # Verify this frame is valid
            is_valid_frame = result.get("is_valid_frame", [])
            if idx >= len(is_valid_frame) or not is_valid_frame[idx]:
                raise ValueError("Invalid frame index - should never happen with good frames")
            
            # Extract natural language goal
            # Around line 1328-1340, replace your current code with:
            is_done_frames = result.get("is_done_frame", [])
            is_done = False
            if idx < len(is_done_frames):
                is_done = is_done_frames[idx]
            # Extract natural language goal


            # Around line 1328-1350, replace your current extraction code with:

            import re

            # Extract natural language goal
            natural_language_goal = result.get("task_description", "Unknown goal")
            task_dict = result.get("task_dict", {})

            # Method 1: Try to extract from synsets (most reliable)
            target_object = None
            extraction_method = None

            # if "synsets" in task_dict and task_dict["synsets"]:
            #     synset = task_dict["synsets"][0]
            #     target_object = synset.split('.')[0]  # 'houseplant' from 'houseplant.n.01'
            #     extraction_method = "synset"

            # Method 2: Fallback to parsing natural language description
            if target_object is None:
                match = re.search(r'(?:find|locate|navigate to|go to|search for)\s+(?:a|an|the)\s+(.+)', natural_language_goal, re.IGNORECASE)
                if match:
                    target_object = match.group(1).strip()
                    extraction_method = "regex"
                else:
                    target_object = natural_language_goal
                    extraction_method = "fallback"


            # Extract unique scene attributes while preserving order
            scene_str = ""
            if include_scene:
                attributes = {
                    "floor_color": {
                        "path": ["floor", "color_name"],
                        "values": [],
                        "seen": set(),
                    },
                    "wall_color": {
                        "path": ["wall", "color_name"],
                        "values": [],
                        "seen": set(),
                    },
                    "room_type": {"path": ["room_type"], "values": [], "seen": set()},
                }
                
                for scene_idx, scene_data in result.get("scene_info", {}).items():
                    if scene_idx <= idx:  # Only consider data seen up to this frame
                        for attr_name, attr_info in attributes.items():
                            # Navigate through the path to get the value
                            value = scene_data
                            for key in attr_info["path"]:
                                if key not in value:
                                    continue
                                value = value[key]
                            
                            if isinstance(value, str) and value not in attr_info["seen"]:
                                if value == "debug hot pink" and "color" in attr_name:
                                    continue  # Skip debug colors
                                attr_info["seen"].add(value)
                                attr_info["values"].append(value)
                
                # Unpack the results
                floor_colors = attributes["floor_color"]["values"]
                wall_colors = attributes["wall_color"]["values"]
                room_types = attributes["room_type"]["values"]
                
                # Construct the scene description
                scene_parts = []
                if floor_colors:
                    scene_parts.append(
                        f"You have already seen floors with the colors: {', '.join(floor_colors)}"
                    )
                if wall_colors:
                    scene_parts.append(
                        f"You have already seen walls with the colors: {', '.join(wall_colors)}"
                    )
                if room_types:
                    scene_parts.append(
                        f"You have been in the room types: {', '.join(room_types)}"
                    )
                
                if scene_parts:
                    scene_str = ". ".join(scene_parts)
                else:
                    scene_str = "You have not observed any specific scene features."
            
            # Extract object information up to this frame
            objects_str = "no objects"
            if include_objects and "selected_objects" in result:
                objects_list = []
                for i, obj in result["selected_objects"].items():
                    # Only include objects encountered before or at the current frame
                    if i <= idx:
                        asset_id = obj.get("asset_id")
                        if asset_id in OBJAVERSE_ANNOTATIONS:
                            obj_short_description = get_shortest_objaverse_option(asset_id)
                        else:
                            obj_lemma = normalize(SYNSET_TO_BEST_LEMMA[obj.get("synset", "")])
                            obj_short_description = f"{obj.get('color_name')} {obj_lemma}"
                        
                        if obj_short_description and obj_short_description not in objects_list:
                            objects_list.append(obj_short_description)
                
                if objects_list:
                    if len(objects_list) == 1:
                        objects_str = objects_list[0]
                    else:
                        # Format list properly with Oxford comma
                        if len(objects_list) == 2:
                            objects_str = f"{objects_list[0]} and {objects_list[1]}"
                        else:
                            objects_str = ", ".join(objects_list[:-1]) + f", and {objects_list[-1]}"
            
            # Construct the final prompt
            prompt_template = (
                "Point to the {goal}\nPlease say 'There are none.' if it is not in the image. If you point to the {goal}, and are close in proximity to the {goal}, "
                "say 'DONE' while pointing to the {goal}. If 'there are none', point to a place on the floor to walk towards so that you explore the house in search of a {goal}. "
            )
            # Add scene and object context if included
            context_parts = []
            if include_scene:
                context_parts.append(f"{scene_str}")
            if include_objects:
                context_parts.append(f"You have been close to {objects_str}")
            
            if context_parts:
                prompt_template += " " + " and ".join(context_parts) + "."
            
            # # Add instruction
            # prompt_template += (
            #     " Point to a point on the floor to walk towards "
            #     "or an object to approach in service of your goal. If you have satisfied "
            #     'your goal, say "DONE" and nothing else.'
            # )
            
            # Add room counting instruction if enabled
            if self.include_room_count:
                prompt_template += " Also, count the number of rooms you think you have seen."
            if self.include_object_reasoning:
                prompt_template = (
                "First enumerate all objects that are currently in the image. Then point to the {goal}\nPlease say 'There are none.' if it is not in the image. If you point to the {goal}, and are close in proximity to the {goal}, "
                "say 'DONE' while pointing to the {goal}. If 'there are none', point to a place on the floor to walk towards so that you explore the house in search of a {goal}. "
            )
            prompt = prompt_template.format(goal=target_object)
            #Add extra negative prompt for each done frame. 
            
            # if is_done:
            #     negative_object = random.choice(ALL_OBJECTS)
            #     prompt_template_negative = (
            #         "Point to {goal}\nPlease say 'There are none.' if it is not in the image. If you point to the {goal}, and are close in proximity to the {goal}, "
            #         "say 'DONE' while pointing to the {goal}."
            #     )
            #     prompt_negative = prompt_template_negative.format(goal=negative_object)
            #     prompts.append(prompt_negative)


            #     prompt_template_negative = (
            #         "Point to {goal}\nPlease say 'There are none.' if it is not in the image. If you point to the {goal}, and are close in proximity to the {goal}, "
            #         "say 'DONE' while pointing to the {goal}."
            #     )
            #     prompt_positive = prompt_template_negative.format(goal=target_object)
            #     prompts.append(prompt_positive)

            # Debug logging for prompts
            if self.debug_mode:
                print(f"DEBUG [_construct_prompts]: Prompt created for object {natural_language_goal} - "
                      f"task_type='{self.task_type}', target_object='{target_object}', "
                      f"prompt='{prompt}'", flush=True)
            
            prompts.append(prompt)
        
        # Return a single string if only one frame was processed, otherwise return the list
        return prompts[0] if single_frame else prompts
    
    def _extract_and_convert_point_coordinates(self, point, camera):
        """
        Helper method to extract coordinates from a point and convert to grid space.
        
        Args:
            point: Point data (dict with x,y or tuple/list)
            camera: Camera name ("front", "right", "left", "down")
        
        Returns:
            tuple: (grid_x, grid_y) in 0-1 grid space, or (None, None) if invalid
        """
        # Extract x, y coordinates (might be different formats)
        if isinstance(point, dict) and "x" in point and "y" in point:
            x = float(point["x"])
            y = float(point["y"])
        elif (isinstance(point, tuple) or isinstance(point, list)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            return None, None
        
        # Convert from camera-space to 2x2 grid-space
        # Grid layout: [front | right]
        #              [left  | down ]
        if camera == "front":
            grid_x = x * 0.5  # 0-0.5 in grid x space
            grid_y = y * 0.5  # 0-0.5 in grid y space
        elif camera == "right":
            grid_x = 0.5 + x * 0.5  # 0.5-1.0 in grid x space
            grid_y = y * 0.5  # 0-0.5 in grid y space
        elif camera == "left":
            grid_x = x * 0.5  # 0-0.5 in grid x space
            grid_y = 0.5 + y * 0.5  # 0.5-1.0 in grid y space
        elif camera == "down":
            grid_x = 0.5 + x * 0.5  # 0.5-1.0 in grid x space
            grid_y = 0.5 + y * 0.5  # 0.5-1.0 in grid y space
        else:
            return None, None
        
        return grid_x, grid_y

    def _construct_labels(self, result, frame_idx):
        """Constructs a label for a given frame."""
        if frame_idx is None or not result or not result.get("valid", False):
            raise ValueError("you have to provide episode information and a valid frame index")
        
        # Handle both single frame and multiple frames
        # Handle both single frame and multiple frames
        single_frame = not isinstance(frame_idx, list)
        frames_to_process = [frame_idx] if single_frame else frame_idx

        labels = []
        objects_strings = []  # ADD THIS LINE - parallel array for objects

        for idx in frames_to_process:
            # Verify this frame is valid
            is_valid_frame = result.get("is_valid_frame", [])
            if idx >= len(is_valid_frame) or not is_valid_frame[idx]:
                raise ValueError("invalid frame index, should not happen with good frames")
            
            # Extract object information for ONLY the current frame
            objects_str = "no objects"
            if "selected_objects" in result:
                objects_list = []
                for i, obj in result["selected_objects"].items():
                    # Only include objects in the CURRENT frame
                    if i == idx:
                        asset_id = obj.get("asset_id")
                        if asset_id in OBJAVERSE_ANNOTATIONS:
                            obj_short_description = get_shortest_objaverse_option(asset_id)
                        else:
                            obj_lemma = normalize(SYNSET_TO_BEST_LEMMA.get(obj.get("synset", ""), ""))
                            obj_short_description = f"{obj.get('color_name')} {obj_lemma}"
                        
                        if obj_short_description and obj_short_description not in objects_list:
                            objects_list.append(obj_short_description)
                
                if objects_list:
                    if len(objects_list) == 1:
                        objects_str = objects_list[0]
                    else:
                        # Format list properly with Oxford comma
                        if len(objects_list) == 2:
                            objects_str = f"{objects_list[0]} and {objects_list[1]}"
                        else:
                            objects_str = ", ".join(objects_list[:-1]) + f", and {objects_list[-1]}"
            
            # Store objects_str for this frame
            objects_strings.append(objects_str)
            
            # ... rest of your existing code that creates labels ...
                    # Check if this is a "DONE" frame
            is_done_frames = result.get("is_done_frame", [])
            is_done = False
            if idx < len(is_done_frames):
                is_done = is_done_frames[idx]
            
            room_count = result["room_count_per_frame"][idx]
            
            # NEW LOGIC: If this is a DONE frame and we want to use object points
            if is_done and self.done_with_object_points:
                # Get object points for this frame
                object_points = result.get("object_points", {}).get(idx, {})
                cameras = ["front", "right", "left", "down"]
                
                # Check if we have object points available
                for camera in cameras:
                    if camera in object_points and object_points[camera]:
                        selected_point = object_points[camera][0]  # Take the first object point
                        
                        # Use helper method for coordinate conversion
                        grid_x, grid_y = self._extract_and_convert_point_coordinates(selected_point, camera)
                        
                        if grid_x is not None and grid_y is not None:
                            # Use object point instead of "DONE"
                            labels.append((True, camera, "object", grid_x, grid_y, 3.0, room_count))
                            break
                else:
                    # No valid object points found, fall back to regular "DONE"
                    labels.append((True, None, None, None, None, None, room_count))
                continue
            elif is_done:
                # Regular DONE handling (existing behavior)
                labels.append((True, None, None, None, None, None, room_count))
                continue
            
            # Get path and object points for this frame
            path_points = result.get("path_points", {}).get(idx, {})
            object_points = result.get("object_points", {}).get(idx, {})
            
            # Prepare to find the most distant point
            max_distance = -1
            selected_point = None
            selected_camera = None
            point_type = None
            
            # Check if we have object points first (priority #1)
            cameras = ["front", "right", "left", "down"]
            has_object_points = False
            for camera in cameras:
                if camera in object_points and object_points[camera]:
                    has_object_points = True
                    # Just take the first object point
                    selected_point = object_points[camera][0]
                    selected_camera = camera
                    point_type = "object"
                    # Assume a default distance for object points
                    max_distance = 3.0  # arbitrary value
                    break
            
            # If we have object points in any camera, we prioritize them
            if has_object_points:
                # We already found our point above
                pass
            # Otherwise, check path points
            elif path_points:
                # Look for cameras with more than one path point
                for camera in cameras:
                    if camera in path_points and len(path_points[camera]) > 1:
                        # This camera has multiple points, find the most distant one
                        for point_idx, point in enumerate(path_points[camera]):
                            if isinstance(point, dict) and "distance" in point:
                                distance = float(point["distance"])
                                if distance > max_distance:
                                    max_distance = distance
                                    selected_point = point
                                    selected_camera = camera
                                    point_type = "path"
            
                # If we didn't find any cameras with multiple points, try cameras with single points
                if selected_point is None:
                    # Count total path points across all cameras
                    total_path_points = sum(
                        len(points)
                        for camera, points in path_points.items()
                        if camera in cameras
                    )
                    
                    # If there's only one path point total across all cameras, skip it (per requirement #2)
                    if total_path_points > 1:
                        # There are multiple path points spread across different cameras
                        # Find the most distant point among all cameras
                        for camera in cameras:
                            if camera in path_points and path_points[camera]:
                                for point_idx, point in enumerate(path_points[camera]):
                                    if isinstance(point, dict) and "distance" in point:
                                        distance = float(point["distance"])
                                        if distance > max_distance:
                                            max_distance = distance
                                            selected_point = point
                                            selected_camera = camera
                                            point_type = "path"
        
            # Create label for this frame
            if selected_point and selected_camera:
                # Use helper method for coordinate conversion
                grid_x, grid_y = self._extract_and_convert_point_coordinates(selected_point, selected_camera)
                
                if grid_x is not None and grid_y is not None:
                    labels.append((False, selected_camera, point_type, grid_x, grid_y, max_distance, room_count))
                else:
                    # If coordinate extraction failed
                    labels.append((False, None, None, None, None, None, room_count))
            else:
                # If no valid points found
                labels.append((False, None, None, None, None, None, room_count))
        
        # Return a single tuple if only one frame was processed, otherwise return the list
        target_actions = []
        for label_idx, label in enumerate(labels):
            is_done, camera, point_type, x, y, distance, room_count = label
            objects_str = objects_strings[label_idx]
            # Format as XML if it's a navigation point (not a "done" state)
            if not is_done and x is not None and y is not None:
                # Scale coordinates from 0-1 to 0-100 and format with 2 decimal places
                x_scaled = format(x * 100, ".2f")
                y_scaled = format(y * 100, ".2f")
                action_xml = f'<point x="{x_scaled}" y="{y_scaled}" alt="go">go</point>'
                if point_type == "path":
                    action_xml = f'There are none. <point x="{x_scaled}" y="{y_scaled}" alt="go">go</point>'
                else:  # point_type == "object"
                    action_xml = f'<point x="{x_scaled}" y="{y_scaled}" alt="go">go</point>'
                target_action = action_xml
            elif is_done and x is not None and y is not None:
                x_scaled = format(x * 100, ".2f")
                y_scaled = format(y * 100, ".2f")
                target_action = f'<point x="{x_scaled}" y="{y_scaled}" alt="DONE">DONE</point>'
            elif is_done and x is None and y is None:
                target_action = "DONE"
            else:
                raise NotImplementedError
                
            # Add room count to the action if enabled
            if self.include_room_count:
                target_action += f". I think I have seen {room_count} rooms so far"
            
            if self.include_room_count:
                target_action += f". I think I have seen {room_count} rooms so far"

            # ADD DEBUG HERE - right before the if statement
            print(f"DEBUG [_construct_labels BEFORE IF]: self.include_object_reasoning = {self.include_object_reasoning}, objects_str = '{objects_str}'", flush=True)
            logger.info(f"DEBUG [_construct_labels BEFORE IF]: self.include_object_reasoning = {self.include_object_reasoning}, objects_str = '{objects_str}'")

            if self.include_object_reasoning:
                print(f"DEBUG [_construct_labels INSIDE IF]: Adding objects to target_action!", flush=True)
                logger.info(f"DEBUG [_construct_labels INSIDE IF]: Adding objects to target_action!")
                target_action = f"I see: {objects_str} in the image. " + target_action
            # Debug logging for labels
            # if self.debug_mode:
            #     print(f"DEBUG [_construct_labels]: Label created - is_done={is_done}, camera={camera}, "
            #           f"point_type={point_type}, coords=({x:.3f if x else None}, {y:.3f if y else None}), "
            #           f"distance={distance:.2f if distance else None}, room_count={room_count}, "
            #           f"target_action='{target_action}'", flush=True)
            # if is_done: 
            #     target_action_negative = "There are none."
            #     target_actions.append(target_action_negative)
            #     #appending positive and negative for curtailed prompt            
            #     target_actions.append(target_action)
            target_actions.append(target_action)
        
        return target_actions[0] if single_frame else target_actions

    def _construct_scene_description_prompt(
        self, result, frame_idx=None, include_scene=True, include_objects=False
    ):
        """
        Constructs a prompt that asks for navigation and a structured scene description.
        Can handle single frame_idx or a list of frame_idx.
        """
        # Get the base prompt(s) (goal, memory)
        base_prompts_or_prompt = self._construct_prompts(result, frame_idx, include_scene, include_objects)

        # Prepare the structured scene description instruction
        description_instruction_intro = "Also, describe the current scene using the following format: "
        format_parts = [
            "Room type: [your observation]",
            "Floor color: [your observation]",
            "Wall color: [your observation]"
        ]
        if include_objects:
            format_parts.append("Nearby objects: [list objects, or 'None']")
        
        description_instruction = description_instruction_intro + "; ".join(format_parts) + "."
        
        # Add room counting instruction if enabled
        
        if self.include_room_count:
            description_instruction += " Also, count the number of rooms you think you have seen."

        if isinstance(base_prompts_or_prompt, list):
            # Multiple base prompts, append instruction to each
            full_prompts = []
            for bp in base_prompts_or_prompt:
                if bp.endswith('.'):
                    full_prompts.append(bp + " " + description_instruction)
                else:
                    full_prompts.append(bp + ". " + description_instruction)
            return full_prompts
        else:
            # Single base prompt
            bp = base_prompts_or_prompt
            if bp.endswith('.'):
                return bp + " " + description_instruction
            else:
                return bp + ". " + description_instruction

    def _construct_scene_description_label(
        self, result, frame_idx=None, include_scene=True, include_objects=False
    ):
        """
        Constructs a label that is either "DONE" or a structured scene description
        combined with the standard action.
        Format if not DONE: "Scene: RoomType: [value]; FloorColor: [value]; ... Action: [standard_action_xml]"
        Can handle single frame_idx or a list of frame_idx.
        """
        if frame_idx is None or not result or not result.get("valid", False):
            raise ValueError("You have to provide episode information and a valid frame index")

        single_frame_input = not isinstance(frame_idx, list)
        frames_to_process = [frame_idx] if single_frame_input else frame_idx
        
        all_combined_labels = []

        for current_single_frame_idx in frames_to_process:
            # First, get the standard action label for the current frame
            # _construct_labels expects a single frame_idx or list, and returns a single label or list.
            # We pass a single frame_idx here, so we expect a single label string.
            standard_action_label = self._construct_labels(result, current_single_frame_idx)

            if "DONE" in standard_action_label:
                all_combined_labels.append(standard_action_label)
                continue

            # If not "DONE", then standard_action_label is an XML point.
            # Now, construct the scene description part.
            
            # Verify this frame is valid (as in _construct_prompts)
            is_valid_frame_list = result.get("is_valid_frame", [])
            if not (0 <= current_single_frame_idx < len(is_valid_frame_list) and is_valid_frame_list[current_single_frame_idx]):
                logger.warning(f"Invalid frame index {current_single_frame_idx} encountered in _construct_scene_description_label for scene part. Skipping.")
                # Fallback or error for this specific frame's combined label
                all_combined_labels.append(f"Scene: Error invalid frame for scene. Action: {standard_action_label}") 
                continue

            scene_desc_parts = []
            current_scene_info_for_frame = {}
            if "scene_info" in result and result["scene_info"]:
                relevant_scene_keys = sorted([
                    k_int for k_int in result["scene_info"].keys() 
                    if isinstance(k_int, int) and k_int <= current_single_frame_idx
                ])
                if relevant_scene_keys:
                    current_scene_info_for_frame = result["scene_info"][relevant_scene_keys[-1]]

            # Room Type
            room_type_val = "Unknown"
            if "room_type" in current_scene_info_for_frame:
                room_type_val = current_scene_info_for_frame["room_type"]
            scene_desc_parts.append(f"Room type: {room_type_val}")

            # Floor Color
            floor_color_val = "Unknown"
            if "floor" in current_scene_info_for_frame and "color_name" in current_scene_info_for_frame["floor"]:
                floor_color_val = current_scene_info_for_frame["floor"]["color_name"]
                if floor_color_val == "debug hot pink": floor_color_val = "Unknown" 
            scene_desc_parts.append(f"Floor color: {floor_color_val}")

            # Wall Color
            wall_color_val = "Unknown"
            if "wall" in current_scene_info_for_frame and "color_name" in current_scene_info_for_frame["wall"]:
                wall_color_val = current_scene_info_for_frame["wall"]["color_name"]
                if wall_color_val == "debug hot pink": wall_color_val = "Unknown"
            scene_desc_parts.append(f"Wall color: {wall_color_val}")

            # Nearby Objects
            if include_objects:
                nearby_objects_list = []
                if "selected_objects" in result:
                    for obj_frame, obj_data in result["selected_objects"].items():
                        if isinstance(obj_frame, int) and obj_frame == current_single_frame_idx:
                            asset_id = obj_data.get("asset_id")
                            obj_short_description = ""
                            if asset_id in OBJAVERSE_ANNOTATIONS:
                                obj_short_description = get_shortest_objaverse_option(asset_id)
                            else:
                                obj_lemma = normalize(SYNSET_TO_BEST_LEMMA.get(obj_data.get("synset", ""), ""))
                                if obj_lemma: 
                                     obj_short_description = f"{obj_data.get('color_name', 'Unknown color')} {obj_lemma}"
                            
                            if obj_short_description and obj_short_description not in nearby_objects_list:
                                # Replace any ';' in object descriptions to avoid breaking the overall structure
                                nearby_objects_list.append(obj_short_description.replace(";", ",")) 
                
                if nearby_objects_list:
                    scene_desc_parts.append(f"Nearby objects: {', '.join(nearby_objects_list)}")
                else:
                    scene_desc_parts.append("Nearby Objects: None")
            
            scene_description_string = "; ".join(scene_desc_parts)
            combined_label = f"Action: {standard_action_label}. Scene: {scene_description_string}"
            
            all_combined_labels.append(combined_label)
        
        return all_combined_labels[0] if single_frame_input else all_combined_labels

    def _get_index_path(self):
        """Get path for the index file."""
        base_name = self._get_cache_file_path_base()
        index_path = os.path.join(
            self.cache_dir, 
            f"{base_name}_shard_index.json"
        )
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        return index_path


    def load_index(self):
        """
        Build or load an index for streaming mode access.
        This method is called during initialization if streaming=True.
        
        Returns:
            The index data structure with information about shards.
        """
        logger.info(f"Building index for {self.task_type_name} ({self.effective_split_for_data_sourcing})")
        
        # Check if there's an existing index file
        index_path = self._get_index_path()
        if os.path.exists(index_path):
            logger.info(f"Found existing index at {index_path}")
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                logger.info(f"Loaded index with {sum(index_data.get('shard_sizes', []))} examples across {len(index_data.get('shard_sizes', []))} shards")
                return index_data
            except Exception as e:
                logger.warning(f"Error loading existing index: {e}. Will rebuild index.")
        
        raise ValueError(f"No index found at {index_path}. This is a problem. Please check your cache directory.")
    
    def _create_error_placeholder(self, idx: int, error_msg: str) -> Dict:
        """Create a placeholder item when there's an error loading data."""
        logger.error(f"Creating error placeholder for index {idx}: {error_msg}")
        return {
            "image": np.zeros((3, 224, 224), dtype=np.uint8),  # Black image
            "prompt": f"Error loading data: {error_msg}",
            "text": "ERROR",
            "style": "pointing",
            "metadata": {
                "image": None,
                "prompt": f"Error loading data: {error_msg}",
                "target_action": "ERROR",
                "episode_id": f"error_{idx}",
                "house_id": "error",
                "frame_idx": -1,
                "task_type": self.task_type,
                "include_scene": self.include_scene,
                "include_objects": self.include_objects,
                "is_done_frame": False,
                "target_box": None,
            },
        }

        

    def _get_shard_and_example_id(self, global_idx):
        """
        Convert a global index to a (shard_id, example_id) pair.
        This maps a sequential global dataset index to the appropriate shard and position within that shard.
        
        Args:
            global_idx: The global index (0-based) in the dataset
            
        Returns:
            Tuple of (shard_id, example_id) representing the shard and position
        """
        if self.data_index is None:
            # If index is missing, load it
            logger.warning(f"Index is missing (why?). Loading it...")
            self.data_index = self.load_index()
        
        if "shard_sizes" not in self.data_index or not self.data_index["shard_sizes"]:
            raise ValueError(f"Invalid index structure: missing or empty shard_sizes")
        
        current_pos = 0
        for shard_idx, shard_size in enumerate(self.data_index["shard_sizes"]):
            if global_idx < current_pos + shard_size:
                # This shard contains the example
                example_id = global_idx - current_pos
                return shard_idx, example_id
            current_pos += shard_size
        
        # If we get here, the index is out of bounds
        raise IndexError(f"Global index {global_idx} out of bounds for dataset with {current_pos} total examples")
    
    def _get_dataset_id(self) -> str:
        """Helper function to get the unique dataset identifier used for temp files, shards, and work assignments."""
        return f"{self.VIDA_SUBPATH}_{self.task_type}_{self.effective_split_for_data_sourcing}_{self.eval_mode_name}_{self.include_scene}_{self.include_objects}_{self.done_with_object_points}_{self.include_room_count}"




if __name__ == "__main__":
    import argparse
    import sys 
    import time
    
    parser = argparse.ArgumentParser(description="Pre-cache robot datasets")
    parser.add_argument("--memory_setting", type=str, default="NoMemory",
                        choices=list(RobotDatasetConfig.MEMORY_SETTINGS.keys()),
                        help="Memory setting for all datasets if --datasets is not used.")
    parser.add_argument("--datasets", nargs="+", 
                        default=None,
                        help="Specific dataset registry keys to cache (e.g., ObjectNavSceneMemory HardObjectNavDoneEvalNoMemory). Overrides --memory_setting and --include_eval if provided.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                        choices=["train", "validation"],
                        help="Which splits to cache for Standard evaluation mode datasets.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count - 1)")
    parser.add_argument("--include_eval", action="store_true", default=False,
                        help="Whether to include DoneEval datasets (uses validation split). Ignored if --datasets is used.")
    parser.add_argument("--overwrite_cache", action="store_true", default=False,
                        help="Whether to overwrite existing cache files.")
    parser.add_argument("--debug_mode", action="store_true", default=False,
                        help="Enable debug mode: limits samples to 5 for all datasets/splits and applies cache limits to train.")
    parser.add_argument("--object_reasoning", action="store_true", default=False,
                        help="Include object reasoning in labels (adds visible objects to action labels).")
    # No CLI argument for prompt_style here, as the cache generated will be style-agnostic.
    # The prompt_style is determined when a dataset *name* is parsed (e.g. from --datasets or in get_dataset_by_name)
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    logging.getLogger("tqdm").setLevel(logging.INFO) 
    logger.setLevel(logging.INFO)

    logger.info(f"Parsed arguments: {args}")
    
    if args.debug_mode:
        logger.warning("DEBUG MODE IS ACTIVE. All datasets will be limited to 5 samples (in-memory and cached), including train splits if applicable. Number of workers will be set to 1.")
    
    debug_sample_limit_value = 5 if args.debug_mode else None

    dataset_configs_to_plan = [] 

    if args.datasets:
        logger.info(f"Processing specific datasets from --datasets: {args.datasets}")
        for reg_key_input in args.datasets:
            original_reg_key = str(reg_key_input)
            processed_key_part = original_reg_key
            parsed_task, parsed_mem, parsed_eval_str = None, None, "Standard"
            parsed_prompt_style = "standard" # Default, this will be passed to constructor
            parsed_done_with_object_points = False  # ADD THIS LINE

            found_mem_key = None
            for mem_key_candidate in sorted(RobotDatasetConfig.MEMORY_SETTINGS.keys(), key=len, reverse=True):
                if mem_key_candidate in processed_key_part:
                    found_mem_key = mem_key_candidate
                    break
            if found_mem_key:
                parsed_mem = found_mem_key
                processed_key_part = processed_key_part.replace(found_mem_key, "", 1) 
            else:
                logger.warning(f"Could not parse memory setting from dataset key '{original_reg_key}'. Skipping.")
                continue

            if "DoneEval" in processed_key_part:
                parsed_eval_str = "DoneEval"
                processed_key_part = processed_key_part.replace("DoneEval", "", 1)
            
            if "SceneDescription" in processed_key_part:
                parsed_prompt_style = "scene_description"
                processed_key_part = processed_key_part.replace("SceneDescription", "", 1)

            # Parse ObjectPointing
            if "ObjectPointing" in processed_key_part:
                parsed_done_with_object_points = True
                processed_key_part = processed_key_part.replace("ObjectPointing", "", 1)

            # Parse RoomCount
            parsed_include_room_count = False
            if "RoomCount" in processed_key_part:
                parsed_include_room_count = True
                processed_key_part = processed_key_part.replace("RoomCount", "", 1)

            found_task_key = None
            if processed_key_part in RobotDatasetConfig.TASK_TYPES:
                found_task_key = processed_key_part
            
            if found_task_key:
                parsed_task = found_task_key
            else:
                logger.warning(f"Could not parse task type from remaining part '{processed_key_part}' of dataset key '{original_reg_key}'. Skipping.")
                continue
            
            splits_for_this_config = ["validation"] if parsed_eval_str == "DoneEval" else args.splits
            for split_arg_for_constructor in splits_for_this_config:
                dataset_configs_to_plan.append({
                    'task_key': parsed_task, 
                    'mem_key': parsed_mem, 
                    'eval_key': parsed_eval_str, 
                    'prompt_style_key': parsed_prompt_style, # This determines instance behavior
                    'split_arg': split_arg_for_constructor, 
                    'original_input': original_reg_key,
                    'done_with_object_points': parsed_done_with_object_points,
                    'include_room_count': parsed_include_room_count
                })
    else: # Default dataset processing (not using --datasets)
        logger.info(f"Processing datasets based on --memory_setting='{args.memory_setting}' and --include_eval={args.include_eval}")
        mem_setting_to_use = args.memory_setting
        default_prompt_style_for_instance = "standard"
        default_done_with_object_points = False
        default_include_room_count = False
        default_include_object_reasoning = args.object_reasoning

        for task_name_key in RobotDatasetConfig.TASK_TYPES:
            for split_arg_for_constructor in args.splits:
                dataset_configs_to_plan.append({
                    'task_key': task_name_key, 
                    'mem_key': mem_setting_to_use, 
                    'eval_key': "Standard", 
                    'prompt_style_key': default_prompt_style_for_instance,
                    'split_arg': split_arg_for_constructor,
                    'original_input': f"{task_name_key}{mem_setting_to_use}",
                    'done_with_object_points': default_done_with_object_points,
                    'include_room_count': default_include_room_count,
                    'include_object_reasoning': default_include_object_reasoning
                })
            if args.include_eval:
                dataset_configs_to_plan.append({
                    'task_key': task_name_key, 
                    'mem_key': mem_setting_to_use, 
                    'eval_key': "DoneEval", 
                    'prompt_style_key': default_prompt_style_for_instance,
                    'split_arg': "validation", 
                    'original_input': f"{task_name_key}DoneEval{mem_setting_to_use}",
                    'done_with_object_points': default_done_with_object_points,
                    'include_room_count': default_include_room_count,
                    'include_object_reasoning': default_include_object_reasoning
                })

    if not dataset_configs_to_plan:
        logger.info("No dataset configurations to process based on the provided arguments. Exiting.")
        sys.exit(0)

    logger.info(f"\n--- PLANNED DATASET PROCESSING ({len(dataset_configs_to_plan)} configurations) ---")
    
    planned_operations = [] 

    for idx, config in enumerate(dataset_configs_to_plan):
        logger.info(
            f"Planning for Config {idx+1}/{len(dataset_configs_to_plan)} "
            f"(Source Input: '{config['original_input']}', Task: {config['task_key']}, Memory: {config['mem_key']}, "
            f"Eval: {config['eval_key']}, PromptStyleForInstance: {config['prompt_style_key']}, SplitArg: '{config['split_arg']}')"
        )
        
        # The prompt_style here sets the default for this instance if loaded.
        # The cache itself will be style-agnostic.
        ds_instance = RobotDataset(
            task_type=config['task_key'],
            split=config['split_arg'], 
            memory_setting=config['mem_key'],
            eval_mode=config['eval_key'],
            prompt_style=config['prompt_style_key'], 
            sample=None, 
            non_train_cache_limit=500, 
            debug_mode_sample_limit=debug_sample_limit_value,
            load_on_init=False,
            done_with_object_points=config.get('done_with_object_points', False),
            include_room_count=config.get('include_room_count', False),
            include_object_reasoning=config.get('include_object_reasoning', False)
        )

        index_path = ds_instance._get_index_path()
        index_exists = os.path.exists(index_path)
        action_description = ""

        if index_exists:
            if args.overwrite_cache:
                action_description = f"Index exists at '{index_path}' and WILL BE REGENERATED."
            else:
                action_description = f"Index exists at '{index_path}'. WILL ATTEMPT TO LOAD FROM IT."
        else:
            action_description = f"Index does NOT exist at '{index_path}'. WILL BE GENERATED."
        
        log_message_for_plan = (
            f"  Instance Details: Task Type Name: {ds_instance.task_type_name}, "
            f"Effective Data Split: {ds_instance.effective_split_for_data_sourcing}, Eval Mode Name: {ds_instance.eval_mode_name}.\n"
            f"  Action: {action_description} Done with Object Points: {ds_instance.done_with_object_points}"
        )
        logger.info(log_message_for_plan)
        
        planned_operations.append({
            'ds_instance': ds_instance,
            'config_details_log': log_message_for_plan, 
        })

    logger.info(f"--- END OF PLAN ({len(planned_operations)} operations) ---")
    
    if not planned_operations:
        logger.info("No operations planned after detailed check. Exiting.")
        sys.exit(0)

    logger.info("Waiting 5 seconds before starting execution...")
    time.sleep(5)

    logger.info("\n--- STARTING EXECUTION OF PLANNED OPERATIONS ---")
    successful_ops = 0
    failed_ops = 0

    for i, op_info in enumerate(planned_operations):
        ds_instance_to_load: RobotDataset = op_info['ds_instance'] 
        
        logger.info(f"\nEXECUTING Operation {i+1}/{len(planned_operations)}:")
        for line in op_info['config_details_log'].splitlines(): 
            logger.info(line)
        
        # Determine the number of workers for this job
        num_workers_for_this_job = args.workers
        if args.debug_mode:
            num_workers_for_this_job = 1
            logger.info("  Debug mode active: Setting number of workers to 1 for this job.")
        
        logger.info(f"  Calling dataset.load() with overwrite_cache={args.overwrite_cache}, num_workers={num_workers_for_this_job}")

        try:
            start_time = time.time()
            
            if args.debug_mode:
                print(f"DEBUG PRINT: __main__ Pre-load for {ds_instance_to_load.task_type_name} (Effective Data Split: {ds_instance_to_load.effective_split_for_data_sourcing}), overwrite={args.overwrite_cache}, workers={num_workers_for_this_job}", flush=True)
            
            ds_instance_to_load.load(
                overwrite_cache=args.overwrite_cache, 
                num_workers=num_workers_for_this_job # Use the determined number of workers
            )
            
            elapsed = time.time() - start_time
            
            if args.debug_mode:
                print(f"DEBUG PRINT: __main__ Post-load for {ds_instance_to_load.task_type_name} (Effective Data Split: {ds_instance_to_load.effective_split_for_data_sourcing}). Actual data len: {len(ds_instance_to_load.data) if ds_instance_to_load.data is not None else 'None'}", flush=True)

            num_loaded_examples = len(ds_instance_to_load.data) if ds_instance_to_load.data is not None else 0
            logger.info(f"  Operation {i+1} COMPLETED in {elapsed:.1f}s. In-memory examples for this instance: {num_loaded_examples}")

            # Instead of inspecting the current instance, create new ones to verify independent access for both prompt styles
            if num_loaded_examples > 0:
                
                for style_to_test in ["standard", "scene_description"]:
                    logger.info(f"  Validating with a fresh dataset instance (prompt_style='{style_to_test}'):")
                    num_samples_to_log = min(3, num_loaded_examples)
                    
                    try:
                        validation_instance = RobotDataset(
                            task_type=ds_instance_to_load.task_type_name,
                            split=ds_instance_to_load.split,
                            memory_setting=[k for k, v in RobotDatasetConfig.MEMORY_SETTINGS.items() 
                                          if v['include_scene'] == ds_instance_to_load.include_scene and 
                                             v['include_objects'] == ds_instance_to_load.include_objects][0],
                            eval_mode=ds_instance_to_load.eval_mode_name,
                            debug_mode_sample_limit=debug_sample_limit_value,
                            load_on_init=True,
                            prompt_style=style_to_test,
                            done_with_object_points=ds_instance_to_load.done_with_object_points,
                            include_room_count=ds_instance_to_load.include_room_count
                        )
                        
                        logger.info(f"    New validation instance (style: {validation_instance.prompt_style}): Task={validation_instance.task_type_name}, "
                                    f"Split={validation_instance.split}, Effective Split={validation_instance.effective_split_for_data_sourcing}, "
                                    f"Eval={validation_instance.eval_mode_name}")
                        
                        if validation_instance.data_index is None:
                            logger.warning(f"    Validation instance (style: {style_to_test}) index is None. Attempting to load from disk...")
                            validation_instance.data_index = validation_instance.load_index()
                        
                        if "shard_sizes" in validation_instance.data_index and "shards" not in validation_instance.data_index:
                            logger.warning(f"    Index structure (style: {style_to_test}) missing 'shards' key. Reconstructing index...")
                            shard_sizes = validation_instance.data_index.get("shard_sizes", [])
                            validation_instance.data_index["shards"] = []
                            for shard_idx_val in range(len(shard_sizes)):
                                shard_path_val = validation_instance._get_shard_path(shard_idx_val)
                                validation_instance.data_index["shards"].append({"path": shard_path_val, "shard_id": shard_idx_val})
                            logger.info(f"    Reconstructed index (style: {style_to_test}) with {len(validation_instance.data_index['shards'])} shards")
                        
                        logger.info(f"    Testing fresh instance (style: {style_to_test}) with {len(validation_instance)} examples")
                        if len(validation_instance) > 0 and num_samples_to_log > 0:
                            random_indices = random.sample(range(len(validation_instance)), min(num_samples_to_log, len(validation_instance)))
                            for sample_idx, data_idx in enumerate(random_indices):
                                try:
                                    example = validation_instance.get(data_idx, None)
                                    prompt = example.get('prompt', 'N/A')
                                    target_action = example.get('text', 'N/A')
                                    
                                    if isinstance(prompt, str) and len(prompt) > 1000: # Shorter truncation for log
                                        prompt = prompt[:1000] + "..."
                                        
                                    logger.info(f"    Sample {sample_idx+1}/{len(random_indices)} (index {data_idx}, style: {style_to_test}): "
                                               f"Prompt='{prompt}', Label='{target_action}'")
                                    
                                    if 'image' in example and example['image'] is not None:
                                        img_shape_info = "Present"
                                        if hasattr(example['image'], 'shape'):
                                            img_shape_info = f"Shape: {example['image'].shape}"
                                        logger.info(f"      Image: {img_shape_info}")
                                        
                                    if 'metadata' in example:
                                        meta_to_log = {k: v for k, v in example['metadata'].items() 
                                                   if k not in ['image', 'prompt', 'target_action']} # Avoid redundant logging
                                        logger.info(f"      Metadata (style: {style_to_test}): {meta_to_log}")
                                        
                                except Exception as access_err:
                                    logger.error(f"      Error accessing example {data_idx} (style: {style_to_test}): {access_err}", exc_info=True)
                        else:
                            logger.info(f"    No examples available to sample for validation (style: {style_to_test}).")
                    
                    except Exception as instance_err:
                        logger.error(f"    Error creating or using validation instance (style: {style_to_test}): {instance_err}", exc_info=True)
                        
            elif num_loaded_examples == 0:
                logger.info(f"  No data examples were loaded/generated for this configuration.")
            
            successful_ops += 1
            
        except Exception as e:
            elapsed = time.time() - start_time
            mem_conf = ds_instance_to_load.memory_config
            mem_str = f"Scene:{mem_conf.get('include_scene', 'UNK')}_Objects:{mem_conf.get('include_objects', 'UNK')}"
            
            logger.error(
                f"  FAILED Operation {i+1} for {ds_instance_to_load.task_type_name} "
                f"(Effective Data Split: {ds_instance_to_load.effective_split_for_data_sourcing}, "
                f"Eval: {ds_instance_to_load.eval_mode_name}, Memory: {mem_str}) after {elapsed:.1f}s: {e}", exc_info=True
            )
            
            failed_ops += 1

    logger.info(f"\n--- ALL PLANNED OPERATIONS PROCESSED ---")
    logger.info(f"Summary: {successful_ops} successful operations, {failed_ops} failed operations.")
    if failed_ops > 0:
        logger.warning("Some operations failed. Please check the logs above for details.")

