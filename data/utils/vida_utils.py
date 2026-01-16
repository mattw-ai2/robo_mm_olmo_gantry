import glob
import json
import os
import random
from typing import Optional

# import cv2
import h5py
import numpy as np
import prior
import torch.multiprocessing as mp
import inspect
import tempfile
import urllib.request
import subprocess

OBJAVERSE_HOME_COMMIT_ID = "ace12898b451c887bb1dd69ede85d32a75a86ef7"


def setup_gitconfig():
    """Create proper gitconfig folder structure to avoid file locking issues."""
    import os
    import subprocess
    
    # Create .gitconfig if it doesn't exist
    home_dir = os.path.expanduser("~")
    gitconfig_path = os.path.join(home_dir, ".gitconfig")
    
    if not os.path.exists(gitconfig_path):
        try:
            with open(gitconfig_path, "w") as f:
                f.write("[filter \"lfs\"]\n")
                f.write("    required = false\n")
                f.write("    smudge = git cat-file blob %f\n")
                f.write("    process = git cat-file blob %f\n")
            print(f"Created gitconfig at {gitconfig_path}")
        except Exception as e:
            print(f"Error creating gitconfig: {e}")


def get_objaverse_annotations(revision=OBJAVERSE_HOME_COMMIT_ID):
    try:
        return prior.load_dataset(
            "objaverse-plus",
            which_dataset="annotations",
            revision=revision,
        )["train"].data
    except Exception as e:
        print(f"Failed to load objaverse annotations: {e}")
        print(f"trying to load from weka instead")
        try:
            with open("/weka/oe-training-default/roseh/backstops/objaverse_annotations_backstop.json", "r") as f:
                return json.load(f)
        except Exception as fallback_e:
            print(f"Failed to load from fallback location: {fallback_e}")
            return {}

OBJAVERSE_ANNOTATIONS = get_objaverse_annotations()

try:
    SYNSET_TO_BEST_LEMMA = prior.load_dataset(
        "objaverse-plus",
        which_dataset="synset_to_best_lemma",
        revision=OBJAVERSE_HOME_COMMIT_ID,
    )["train"].data
except Exception as e:
    print(f"Failed to load synset_to_best_lemma: {e}")
    print(f"Trying to load from weka instead")
    try:
        with open("/weka/oe-training-default/roseh/backstops/synset_to_best_lemma_backstop.json", "r") as f:
            SYNSET_TO_BEST_LEMMA = json.load(f)
    except Exception as fallback_e:
        print(f"Failed to load from fallback location: {fallback_e}")
        print("Using empty fallback.")
        SYNSET_TO_BEST_LEMMA = {}


def convert_byte_to_string(bytes_to_decode: np.ndarray, max_len: Optional[int] = None):
    if max_len is None:
        max_len = bytes_to_decode.shape[-1]
    return (bytes_to_decode.view(f"S{max_len}")[0]).decode()


def normalize(text):
    # if ".n." in text:
    #     return (
    #         best_lemma(text, precomputed=True).lower().replace("_", " ").strip().strip(".;/,'\"\\")
    #     )
    # else:
    return text.strip().lower().replace("_", " ").strip().strip(".;/,'\"\\")


def clean_description(desc):
    if desc is None:
        return ""
    desc = normalize(desc)

    # handler for `this is worn on your feet to allow support when you walk and run.`
    if desc.startswith("this is worn"):
        desc = "the thing that" + desc[len("this") :]

    to_remove_in_order = [
        "i think this might be",
        "it looks to be",
        "it looks like",
        "it looks almost like",
        "it looks more like",
        "it look like",
        "this looks like",
        "this appears to be",
        "it it",
        "it is",
        "it's",
        "it s",
        "this is",
        "these are",
        "it looks",
        "it could be",
        "it",
        "there is",
        "there are",
        "these look like",
        "actually",
        "here are",
    ]

    for prefix in to_remove_in_order:
        if desc.startswith(prefix):
            desc = normalize(desc[len(prefix) :])

    # `they` only appears as a typo in starting position?
    if desc.split()[0] in [
        "a",
        "an",
        "some",
        "they",
        "and",
        "am",
        "as",
        "sa",
        "various",
    ]:
        desc = " ".join(["the"] + desc.split()[1:])

    # handler for `this connects to your computer and is used to type`
    if desc.startswith("this connects"):
        desc = "the thing that" + desc[len("this") :]

    # handler for a few remaining descriptions starting with `this`
    if desc.startswith("this"):
        desc = "the" + desc[len("this") :]

    # `He has long toes on his feet with a plaid backpack and plaid hat`
    if desc.startswith("he"):
        desc = "the man" + desc[len("he") :]

    if desc.startswith("i"):
        desc = "the" + desc[len("i") :]

    # `what remains of ...` doesn't need `the`, nor seeking for verbs
    if desc.split()[0] not in ["the", "what"]:
        desc = f"the {desc}"

    if desc.split()[0] == "the":
        words = desc.split()
        constraint_seen = False
        for it, word in enumerate(words):
            if word in ["is", "are", "shows", "contains", "has", "have"]:
                if not constraint_seen:
                    desc = " ".join(words[:it] + [f"that {word}"] + words[it + 1 :])
                break
            elif (
                word in ["that", "who", "with", "which", "where", "but", "and"]
                or "." in word
                or "," in word
                or ";" in word
            ):
                # Note: adding `and` breaks a handful of sentences like `go to the black and white book has stripes`,
                # but also handles some ungrammatical descriptions. Humans also make a lot of typos, anyway.
                constraint_seen = True
            elif word in ["that's", "thats"]:
                break

    desc = normalize(desc)

    # Remove trailing period
    if desc[-1] == ".":
        desc = desc[:-1]

    return desc


def get_shortest_objaverse_option(asset_id):
    ann = OBJAVERSE_ANNOTATIONS[asset_id]
    all_descriptions = [ann["description"], ann["description_auto"], ann["category"]]
    all_descriptions = [clean_description(desc) for desc in all_descriptions]
    # actually, pick the longest that's shorter than 60 characters
    all_descriptions = [
        desc for desc in all_descriptions if len(desc) < 60
    ]  # functionally backstopped by category, hopefully
    return max(all_descriptions, key=len)


def parse_local_scene_info(local_scene_data) -> str:
    """Parse LocalSceneDescription sensor data into human-readable text."""
    if local_scene_data is None or len(local_scene_data) == 0:
        return "No LocalSceneDescription data available"

    try:
        # Try to convert to string if it's bytes
        if isinstance(local_scene_data, (bytes, np.ndarray)):
            local_scene_str = convert_byte_to_string(local_scene_data)
        else:
            local_scene_str = str(local_scene_data)

        # Try to parse as JSON if possible
        try:
            local_scene_dict = json.loads(local_scene_str)
            # Format the dictionary as a pretty string
            return json.dumps(local_scene_dict, indent=2)
        except:
            # If not JSON, return as is
            return local_scene_str
    except Exception as e:
        return f"Error parsing LocalSceneDescription: {str(e)}"


def json_templated_spec_to_dict(task_spec_string):
    """Convert a JSON task specification string to a dictionary."""
    try:
        return json.loads(task_spec_string)
    except:
        return {}


def find_random_hdf5_file(dataset_path: str, subset_type: str = "") -> str:
    """Find a random HDF5 file in the dataset."""
    if subset_type:
        search_path = os.path.join(dataset_path, subset_type, "**/*.hdf5")
    else:
        search_path = os.path.join(dataset_path, "**/*.hdf5")

    hdf5_files = glob.glob(search_path, recursive=True)

    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {dataset_path}")

    return random.choice(hdf5_files)


def has_valid_points(episode_data, frame_idx):
    """Check if a frame has valid (non-empty) future path or object points for at least one camera.

    Returns:
        tuple: (has_valid_points, parsed_path_data, parsed_object_data)
    """
    has_path_points = False
    has_object_points = False
    parsed_path_data = {}
    parsed_object_data = {}

    # Check future path points
    if "future_path_image_points" in episode_data and frame_idx < len(
        episode_data["future_path_image_points"]
    ):
        try:
            future_path_data = json.loads(
                convert_byte_to_string(
                    episode_data["future_path_image_points"][frame_idx]
                )
            )
            parsed_path_data = future_path_data

            for camera in ["front", "right", "left", "down"]:
                if (
                    camera in future_path_data
                    and isinstance(future_path_data[camera], list)
                    and len(future_path_data[camera]) > 0
                ):
                    has_path_points = True
                    break
        except Exception as e:
            # print(f"Error checking future path points: {e}")
            # Initialize empty path data on error
            parsed_path_data = {"front": [], "right": [], "left": [], "down": []}

    # Check future object points
    if "future_object_image_points" in episode_data and frame_idx < len(
        episode_data["future_object_image_points"]
    ):
        try:
            future_object_data = json.loads(
                convert_byte_to_string(
                    episode_data["future_object_image_points"][frame_idx]
                )
            )
            parsed_object_data = future_object_data

            for camera in ["front", "right", "left", "down"]:
                if (
                    camera in future_object_data
                    and isinstance(future_object_data[camera], list)
                    and len(future_object_data[camera]) > 0
                ):
                    has_object_points = True
                    break
        except Exception as e:
            print(f"Error checking future object points: {e}")
            # Initialize empty object data on error
            parsed_object_data = {"front": [], "right": [], "left": [], "down": []}

    return (has_path_points or has_object_points), parsed_path_data, parsed_object_data


def parse_episode_data(episode_data):
    """
    Couple logic points here:
    1. this episode has any frames with future path or object points
    2. none of the floors or walls are "debug hot pink" before a frame - once this color appears, all subsequent frames are invalid
    3. it is longer than 10 frames (necessary?)

    this returns a boolean for the episode validity, path and object points per frame, and the floor and wall colors at every step. this is formatted as a dict.
    """
    # Find the length of the episode
    sequence_length = None
    for key in episode_data.keys():
        if (
            isinstance(episode_data[key], h5py.Dataset)
            and len(episode_data[key].shape) > 0
        ):
            sequence_length = episode_data[key].shape[0]
            break

    # Check if the episode is long enough
    if sequence_length is None or sequence_length < 10:
        return False, {}, {}

    # Check for valid points and collect path/object data
    has_any_valid_points = False
    path_points_per_frame = {}
    object_points_per_frame = {}
    scene_info = {}
    nearby_objects_per_frame = set()
    selected_objects_info = {}  # Store full object information
    is_valid_frame = [False] * sequence_length  # Boolean list for valid frames
    is_done_frame = [False] * sequence_length  # Initialize as list of False values
    room_count_per_frame = [None] * sequence_length  # Initialize as list of 0 values
    last_done_frame_idx = None  # Track the last frame index where done was True

    # Flag to track if debug hot pink has been encountered
    encountered_debug_hot_pink = False

    for frame_idx in range(sequence_length):
        # Check for valid points in this frame
        has_points, path_data, object_data = has_valid_points(episode_data, frame_idx)

        has_success = bool(episode_data["hypothetical_task_success"][frame_idx])
        #is_done_frame[frame_idx] = has_success  # Store in the boolean list
        last_done_frame_idx = frame_idx if has_success else last_done_frame_idx

        local_scene_info = episode_data["local_scene_info"][frame_idx]
        local_scene_info = json.loads(parse_local_scene_info(local_scene_info))

        # Extract floor and wall colors
        temp_scene_info = {
            k: (
                {
                    k2: v2
                    for k2, v2 in v.items()
                    if not (k2 == "type" and v2 == "PLACEHOLDER")
                }
                if isinstance(v, dict)
                else v
            )
            for k, v in local_scene_info["scene"].items()
        }
        if (
            temp_scene_info["floor"]["color_name"] == "debug hot pink"
            or temp_scene_info["wall"]["color_name"] == "debug hot pink"
        ):
            encountered_debug_hot_pink = True
        
        # Extract room count
        room_count_per_frame[frame_idx] = episode_data["rooms_seen"][frame_idx] + 1 # in vida repo this sensor is technically "rooms seen and then left" so off by one

        # Extract nearby objects that are different from the current set, and select a single one
        if not encountered_debug_hot_pink:
            scene_info[frame_idx] = temp_scene_info
            unique_nearby_objects = [
                obj
                for obj in local_scene_info["objects"]
                if obj["asset_id"] not in nearby_objects_per_frame
            ]
            if unique_nearby_objects:
                selected_object = random.choice(unique_nearby_objects)
                nearby_objects_per_frame.add(selected_object["asset_id"])
                selected_objects_info[frame_idx] = (
                    selected_object  # Store full object info
                )
            else:
                # print(f"No unique nearby objects found for frame {frame_idx}")
                pass

        # Add frame to valid_frames if it has points or success flag and we haven't seen debug hot pink yet
        if (has_points or has_success) and not encountered_debug_hot_pink:
            if not has_any_valid_points:
                has_any_valid_points = has_points
            is_valid_frame[frame_idx] = True  # Mark this frame as valid

            # Store points data
            path_points_per_frame[frame_idx] = path_data
            object_points_per_frame[frame_idx] = object_data
        else:
            pass
            # print(f"Frame {frame_idx} is invalid. Bools are: has points {has_points}, has success {has_success}, encountered debug hot pink {encountered_debug_hot_pink}")
            
        if last_done_frame_idx is not None:
            is_done_frame[last_done_frame_idx] = True
        
    if sum(is_valid_frame) > 5:
        # Extract task information
        task_dict = {}
        task_description = ""
        if (
            "templated_task_spec" in episode_data
            and len(episode_data["templated_task_spec"]) > 0
        ):
            try:
                task_spec = episode_data["templated_task_spec"][
                    -1
                ]  # Usually the last one is the complete spec
                task_spec_str = convert_byte_to_string(task_spec, None)
                task_dict = json_templated_spec_to_dict(task_spec_str)
                task_type = task_dict.get("task_type", "Unknown")
                task_description = task_dict["extras"]["natural_language_description"]
            except Exception:
                pass

        # Compile results
        result = {
            "valid": has_any_valid_points,
            "path_points": path_points_per_frame,
            "object_points": object_points_per_frame,
            "scene_info": scene_info,
            "selected_objects": selected_objects_info,  # Add full object information
            "sequence_length": sequence_length,
            "is_valid_frame": is_valid_frame,  # List of booleans for valid frames
            "task_dict": task_dict,
            "task_description": task_description,
            "is_done_frame": is_done_frame,  # List of booleans
            "room_count_per_frame": room_count_per_frame,  # List of room counts
        }
    else:
        result = {}

    return has_any_valid_points, result, {}


def select_good_frames(result, max_to_return=10, final_frame_only=False):
    """
    Selects multiple good frames from an episode based on specific criteria.

    When final_frame_only=True, only returns the final "DONE" frame.
    Otherwise, "Good" frames are defined in this priority order:
    1. "Done" frames (task completed)
    2. Frames with path points > 4 meters distance
    3. Frames with both object points and path points

    If not enough good frames are found, we fall back to:
    4. Frames with maximum distance path points
    5. Later frames in the episode

    All good frames go into a pool from which we select non-sequential frames.

    Args:
        result (dict): Dictionary containing episode information from parse_episode_data
        max_to_return (int): Maximum number of frames to return
        final_frame_only (bool): If True, only return the final frame where is_done_frame is True

    Returns:
        list: List of frame indices, or empty list if no valid frames
    """
    if not result or not result.get("valid", False):
        return []

    # Get all valid frame indices from the boolean list
    is_valid_frame = result.get("is_valid_frame", [])
    valid_frames = [idx for idx, is_valid in enumerate(is_valid_frame) if is_valid]

    if not valid_frames:
        return []
    
    # keep max_to_return proportional to episode length
    # max_to_return = max(max_to_return, int(len(valid_frames) // 20))  # at least 5, or 5% of valid frames
        
    # If we only want the final "DONE" frame for DONE evaluation
    if final_frame_only:
        is_done_frames = result.get("is_done_frame", [])
        done_frames = [
            idx for idx in valid_frames 
            if idx < len(is_done_frames) and is_done_frames[idx]
        ]
        
        # If there are any done frames, return the last one (final success state)
        if done_frames:
            return [done_frames[-1]]
        return []

    # Initialize our pool of good frames
    good_frames_pool = set()  # Using a set to avoid duplicates

    # 1. Find "done" frames - these are automatically considered good frames
    is_done_frames = result.get("is_done_frame", [])
    done_frames = [
        idx for idx in valid_frames if idx < len(is_done_frames) and is_done_frames[idx]
    ]

    # Add all done frames to our pool of good frames
    if len(done_frames) > 1:
        # good_frames_pool.update([random.choice(done_frames)])
        # half of the frames can be from done_frames
        num_to_choose = min(max_to_return // 2, len(done_frames))
        chosen_done_frames = random.sample(done_frames, num_to_choose)
        good_frames_pool.update(chosen_done_frames)

    # Track frames by different criteria
    frames_with_distant_paths = []  # Distance > 4m
    frames_with_both_types = []  # Both path and object points
    frames_by_max_distance = []  # Sorted by max distance

    # Analyze all valid frames
    for frame_idx in valid_frames:
        # Skip frames already in the pool (done frames)
        if frame_idx in good_frames_pool:
            continue

        # Check path points
        path_points = result.get("path_points", {}).get(frame_idx, {})
        has_path_points = False
        max_path_distance = 0
        has_distant_points = False  # > 4m

        for camera in ["front", "right", "left", "down"]:
            if camera in path_points and path_points[camera]:
                points = path_points[camera]
                if points:
                    has_path_points = True

                # Find max distance for this camera
                for point in points:
                    if isinstance(point, dict) and "distance" in point:
                        distance = float(point["distance"])
                        max_path_distance = max(max_path_distance, distance)
                        if distance > 4.0:  # Threshold for "distant" points
                            has_distant_points = True

        # Check object points
        object_points = result.get("object_points", {}).get(frame_idx, {})
        has_object_points = False

        for camera in ["front", "right", "left", "down"]:
            if camera in object_points and object_points[camera]:
                if object_points[camera]:
                    has_object_points = True
                    break

        # 2. Frames with path points > 4 meters distance
        if has_distant_points:
            frames_with_distant_paths.append(frame_idx)

        # 3. Frames with both object points and path points
        if has_path_points and has_object_points:
            frames_with_both_types.append(frame_idx)

        # 4. Keep track of all frames with path points for sorting by distance
        if has_path_points and max_path_distance > 0:
            frames_by_max_distance.append((frame_idx, max_path_distance))

    # Add frames to the pool based on criteria 2: distant path points
    good_frames_pool.update(frames_with_distant_paths)

    # Add frames to the pool based on criteria 3: both object and path points
    good_frames_pool.update(frames_with_both_types)

    # If we still don't have enough frames, add frames with maximum distance path points
    if len(good_frames_pool) < max_to_return and frames_by_max_distance:
        # Sort by distance, descending
        frames_by_max_distance.sort(key=lambda x: x[1], reverse=True)

        # Add frames until we reach our target, but only if distance >= 2 meters
        for frame_idx, distance in frames_by_max_distance:
            if (
                frame_idx not in good_frames_pool and distance >= 2.0
            ):  # Added minimum distance check
                good_frames_pool.add(frame_idx)
                if len(good_frames_pool) >= max_to_return:
                    break

    # If we still don't have enough frames, add later frames in the episode
    if len(good_frames_pool) < max_to_return:
        # Sort valid frames by index (later frames first)
        remaining_frames = [idx for idx in valid_frames if idx not in good_frames_pool]
        remaining_frames.sort(reverse=True)

        # Add frames until we reach our target
        for frame_idx in remaining_frames:
            good_frames_pool.add(frame_idx)
            if len(good_frames_pool) >= max_to_return:
                break

    # Convert set back to list and sort
    good_frames_pool = sorted(list(good_frames_pool))

    # Select non-sequential frames
    selected_frames = []

    # If we have too few frames to worry about adjacency
    if len(good_frames_pool) <= max_to_return:
        # Special case: If we have sequential frames but too few to filter
        # Try to keep at least some frames by removing adjacents
        i = 0
        while i < len(good_frames_pool) and len(selected_frames) < max_to_return:
            selected_frames.append(good_frames_pool[i])
            i += 2  # Skip the next frame to avoid adjacency
    else:
        # Use a greedy approach to select non-sequential frames
        remaining_candidates = good_frames_pool.copy()

        while remaining_candidates and len(selected_frames) < max_to_return:
            # Randomly select a frame from remaining candidates
            selected_frame = random.choice(remaining_candidates)
            selected_frames.append(selected_frame)

            # Remove the selected frame and its adjacent frames
            remaining_candidates = [
                f for f in remaining_candidates if abs(f - selected_frame) > 1
            ]

    return selected_frames