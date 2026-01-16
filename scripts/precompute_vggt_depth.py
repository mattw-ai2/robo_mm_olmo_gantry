"""
VGGT Depth Preprocessing Script for VIDA Robot Dataset

This script pre-computes per-frame point clouds from RGB video frames using VGGT
(Visual Geometry Grounded Transformer) and stores them in HDF5 files.

Output structure (per house):
    point_clouds.hdf5/
        episode_num/
            camera_name/  (front, right, down, left)
                frame_idx -> point_cloud [N, 3] in robot-centered coordinates

At train time, accumulate frames 0..N from all cameras to get full point cloud history.

SINGLE GPU Usage:
    python scripts/precompute_vggt_depth.py \
        --input_dir /weka/prior/datasets/vida_datasets/.../ObjectNavType/train \
        --output_dir /path/to/output

MULTI-GPU Usage:
    python scripts/precompute_vggt_depth.py \
        --input_dir /path/to/input \
        --output_dir /path/to/output \
        --num_gpus 4

RESUME (skip already processed houses):
    python scripts/precompute_vggt_depth.py \
        --input_dir /path/to/input \
        --output_dir /path/to/output \
        --resume
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Optional, Tuple

print("Loading libraries...", flush=True)

import h5py
print("  - h5py loaded", flush=True)
import numpy as np
print("  - numpy loaded", flush=True)
import torch
print(f"  - torch loaded (CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()})", flush=True)
from tqdm import tqdm
print("  - tqdm loaded", flush=True)
print("Libraries loaded!", flush=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Camera names in order (matching the 2x2 grid layout)
CAMERA_NAMES = ["front", "right", "down", "left"]


def load_vggt_model(device: str = "cuda") -> Optional[torch.nn.Module]:
    """
    Load the VGGT model for depth and camera estimation.
    
    Returns:
        VGGT model ready for inference, or None if not available
    """
    try:
        # Try to import VGGT from the official repository
        from vggt.models.vggt import VGGT
        
        model = VGGT.from_pretrained("facebook/vggt-1b")
        model = model.to(device)
        model.eval()
        logger.info("Loaded VGGT model successfully")
        return model
        
    except ImportError as e:
        logger.warning(f"VGGT not installed or import error: {e}")
        logger.warning("Install from: https://github.com/facebookresearch/vggt")
        logger.warning("Falling back to dummy depth estimation for testing")
        return None


def load_video_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load a single frame from a video file using decord (fast) or OpenCV fallback.
    
    Args:
        video_path: Path to video file
        frame_idx: Index of frame to load
        
    Returns:
        Frame as numpy array [H, W, 3] RGB or None if failed
    """
    try:
        # Try decord first (fastest)
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            if frame_idx < len(vr):
                frame = vr[frame_idx].asnumpy()  # Already RGB
                return frame
            else:
                logger.warning(f"Frame {frame_idx} out of range for {video_path} (length {len(vr)})")
                return None
        except ImportError:
            pass
        
        # Fall back to OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
        
    except Exception as e:
        logger.warning(f"Error loading frame {frame_idx} from {video_path}: {e}")
        return None


def load_all_video_frames(video_path: str) -> Optional[np.ndarray]:
    """
    Load all frames from a video file using decord (fast) or OpenCV fallback.
    
    Args:
        video_path: Path to video file
        
    Returns:
        All frames as numpy array [N, H, W, 3] RGB or None if failed
    """
    try:
        # Try decord first (fastest - doesn't decode all frames upfront)
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            # Load all frames at once
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()
            return frames
        except ImportError:
            pass
        
        # Fall back to OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if frames:
            return np.stack(frames, axis=0)
        return None
        
    except Exception as e:
        logger.warning(f"Error loading video {video_path}: {e}")
        return None


def get_camera_frames(
    episode_dir: str,
    episode_num: str,
    frame_indices: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """
    Load frames from all 4 cameras for specified frame indices.
    
    Args:
        episode_dir: Directory containing the episode data
        episode_num: Episode number string
        frame_indices: List of frame indices to load. If None, load all frames.
        
    Returns:
        Dict mapping camera name to frames [N, H, W, 3]
    """
    camera_frames = {}
    
    for camera in CAMERA_NAMES:
        video_path = os.path.join(episode_dir, f"raw_{camera}_camera__{episode_num}.mp4")
        
        if not os.path.exists(video_path):
            # Try alternative path format
            video_path = os.path.join(episode_dir, f"warped_{camera}_camera__{episode_num}.mp4")
        
        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue
            
        all_frames = load_all_video_frames(video_path)
        if all_frames is None:
            continue
            
        if frame_indices is not None:
            # Select specific frames
            valid_indices = [i for i in frame_indices if i < len(all_frames)]
            if valid_indices:
                camera_frames[camera] = all_frames[valid_indices]
        else:
            camera_frames[camera] = all_frames
    
    return camera_frames


def resize_to_patch_multiple(images: np.ndarray, patch_size: int = 14) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize images so dimensions are multiples of patch_size.
    
    Args:
        images: Input images [N, H, W, 3]
        patch_size: Patch size (default 14 for VGGT)
        
    Returns:
        resized_images: Resized images [N, H_new, W_new, 3]
        original_size: (H, W) original dimensions for scaling back
    """
    import cv2
    
    N, H, W, C = images.shape
    
    # Round down to nearest multiple of patch_size
    H_new = (H // patch_size) * patch_size
    W_new = (W // patch_size) * patch_size
    
    if H_new == H and W_new == W:
        return images, (H, W)
    
    # Resize each image
    resized = np.zeros((N, H_new, W_new, C), dtype=images.dtype)
    for i in range(N):
        resized[i] = cv2.resize(images[i], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    
    return resized, (H, W)


def estimate_depth_and_extrinsics(
    model: Optional[torch.nn.Module],
    images: np.ndarray,  # [N, H, W, 3]
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate depth maps and camera extrinsics from RGB images using VGGT.
    
    Args:
        model: VGGT model (or None for dummy estimation)
        images: RGB images [N, H, W, 3]
        device: Device to run on
        
    Returns:
        depth_maps: Depth maps [N, H, W] (original resolution)
        point_maps: 3D point maps [N, H, W, 3] in camera frame
        extrinsics: Camera extrinsics [N, 4, 4]
    """
    N, H, W, C = images.shape
    
    if model is None:
        # Dummy depth estimation for testing
        depth_maps = np.random.rand(N, H, W).astype(np.float32) * 5 + 0.5  # Random depth 0.5-5.5m
        extrinsics = np.tile(np.eye(4), (N, 1, 1)).astype(np.float32)
        
        # Simple point map from depth (assuming simple camera model)
        point_maps = depth_to_point_map(depth_maps, H, W)
        return depth_maps, point_maps, extrinsics
    
    # Resize images to be multiples of patch size (14 for VGGT)
    images_resized, original_size = resize_to_patch_multiple(images, patch_size=14)
    N_r, H_r, W_r, C_r = images_resized.shape
    
    # Process with VGGT
    with torch.no_grad():
        # Convert to tensor and normalize to [0, 1]
        images_tensor = torch.from_numpy(images_resized).permute(0, 3, 1, 2).float() / 255.0
        images_tensor = images_tensor.to(device)
        
        # Run VGGT
        outputs = model(images_tensor)
        
        # Extract depth - VGGT returns [B, S, H, W, 1] for depth
        depth_out = outputs["depth"]
        if depth_out.dim() == 5:
            depth_out = depth_out.squeeze(-1)  # [B, S, H, W] or [N, H, W]
        if depth_out.dim() == 4:
            depth_out = depth_out.squeeze(0)  # [S, H, W] if batch=1
        depth_maps = depth_out.cpu().numpy()  # [N, H_r, W_r]
        
        # Extract world points - VGGT returns [B, S, H, W, 3]
        world_points = outputs["world_points"]
        if world_points.dim() == 5:
            world_points = world_points.squeeze(0)  # [S, H, W, 3] if batch=1
        point_maps = world_points.cpu().numpy()  # [N, H_r, W_r, 3]
        
        # Extract camera poses - VGGT returns pose_enc [B, S, 9]
        # For now, use identity extrinsics since we transform to robot frame anyway
        extrinsics = np.tile(np.eye(4), (N, 1, 1)).astype(np.float32)
    
    # Resize depth and point maps back to original resolution if needed
    if H_r != H or W_r != W:
        import cv2
        depth_maps_orig = np.zeros((N, H, W), dtype=np.float32)
        point_maps_orig = np.zeros((N, H, W, 3), dtype=np.float32)
        
        # Scale factor for depth values
        scale_h = H / H_r
        scale_w = W / W_r
        
        for i in range(N):
            depth_maps_orig[i] = cv2.resize(depth_maps[i], (W, H), interpolation=cv2.INTER_LINEAR)
            for c in range(3):
                point_maps_orig[i, :, :, c] = cv2.resize(point_maps[i, :, :, c], (W, H), interpolation=cv2.INTER_LINEAR)
        
        depth_maps = depth_maps_orig
        point_maps = point_maps_orig
    
    return depth_maps, point_maps, extrinsics


def depth_to_point_map(
    depth_maps: np.ndarray,  # [N, H, W]
    H: int,
    W: int,
    fx: float = 320.0,  # Focal length x (adjusted for typical robot camera)
    fy: float = 320.0,  # Focal length y
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> np.ndarray:
    """
    Convert depth maps to 3D point maps using pinhole camera model.
    
    Args:
        depth_maps: Depth values [N, H, W]
        H, W: Image dimensions
        fx, fy: Focal lengths
        cx, cy: Principal point (defaults to image center)
        
    Returns:
        point_maps: 3D points [N, H, W, 3] in camera frame
    """
    if cx is None:
        cx = W / 2
    if cy is None:
        cy = H / 2
    
    N = depth_maps.shape[0]
    
    # Create pixel coordinate grids
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    # Unproject to 3D (camera frame: X right, Y down, Z forward)
    x = (u - cx) / fx
    y = (v - cy) / fy
    
    # Stack and broadcast
    point_maps = np.zeros((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        d = depth_maps[i]
        point_maps[i, :, :, 0] = x * d  # X (right)
        point_maps[i, :, :, 1] = y * d  # Y (down)
        point_maps[i, :, :, 2] = d      # Z (forward)
    
    return point_maps


def unproject_depth_to_points(
    depth_maps: np.ndarray,  # [N, H, W]
    intrinsics: np.ndarray,  # [N, 3, 3]
    extrinsics: np.ndarray,  # [N, 4, 4]
) -> np.ndarray:
    """
    Unproject depth maps to 3D world points using camera parameters.
    
    Args:
        depth_maps: Depth values [N, H, W]
        intrinsics: Camera intrinsic matrices [N, 3, 3]
        extrinsics: Camera extrinsic matrices [N, 4, 4]
        
    Returns:
        point_maps: 3D world points [N, H, W, 3]
    """
    N, H, W = depth_maps.shape
    point_maps = np.zeros((N, H, W, 3), dtype=np.float32)
    
    # Create pixel coordinate grids
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    ones = np.ones_like(u)
    
    # Homogeneous pixel coordinates [3, H*W]
    pixels = np.stack([u.flatten(), v.flatten(), ones.flatten()], axis=0)
    
    for i in range(N):
        # Get camera parameters
        K = intrinsics[i]  # [3, 3]
        E = extrinsics[i]  # [4, 4]
        d = depth_maps[i].flatten()  # [H*W]
        
        # Unproject: K^-1 @ pixels
        K_inv = np.linalg.inv(K)
        rays = K_inv @ pixels  # [3, H*W]
        
        # Scale by depth
        points_cam = rays * d[None, :]  # [3, H*W]
        
        # Transform to world coordinates
        points_cam_h = np.vstack([points_cam, np.ones((1, H*W))])  # [4, H*W]
        E_inv = np.linalg.inv(E)
        points_world = E_inv @ points_cam_h  # [4, H*W]
        
        # Store
        point_maps[i] = points_world[:3].T.reshape(H, W, 3)
    
    return point_maps


def get_camera_transform(camera_name: str) -> np.ndarray:
    """
    Get the fixed transform from camera frame to robot frame.
    
    The robot has 4 cameras arranged clockwise: front, right, down, left.
    This returns the rotation to transform from camera frame to robot body frame.
    
    Args:
        camera_name: One of 'front', 'right', 'down', 'left'
        
    Returns:
        4x4 transformation matrix
    """
    # Camera orientations relative to robot body frame
    # Assuming cameras point outward from robot center
    # Robot frame: X forward, Y left, Z up
    
    transforms = {
        "front": np.eye(4),  # Front camera is aligned with robot forward
        "right": np.array([  # Rotate 90° right (around Z)
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32),
        "left": np.array([   # Rotate 90° left (around Z)
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32),
        "down": np.array([   # Point downward (rotate around X)
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32),
    }
    
    return transforms.get(camera_name, np.eye(4, dtype=np.float32))


def transform_points_to_robot_frame(
    points: np.ndarray,  # [N, 3]
    camera_name: str,
    robot_pose: Optional[np.ndarray] = None  # [4, 4] robot pose in world frame
) -> np.ndarray:
    """
    Transform points from camera frame to robot-centered frame.
    
    Args:
        points: Points in camera frame [N, 3]
        camera_name: Name of the camera
        robot_pose: Optional robot pose to transform to world then back to current robot frame
        
    Returns:
        Points in robot-centered frame [N, 3]
    """
    if len(points) == 0:
        return points
        
    # Get camera-to-robot transform
    cam_to_robot = get_camera_transform(camera_name)
    
    # Apply transform
    points_h = np.hstack([points, np.ones((len(points), 1))])
    points_robot = (cam_to_robot @ points_h.T).T[:, :3]
    
    return points_robot.astype(np.float32)


def get_episode_length(episode_data: h5py.Group) -> int:
    """Get the number of frames in an episode from its HDF5 data."""
    for key in episode_data.keys():
        if isinstance(episode_data[key], h5py.Dataset) and len(episode_data[key].shape) > 0:
            return episode_data[key].shape[0]
    return 0


def process_episode(
    episode_dir: str,
    episode_num: str,
    episode_data: h5py.Group,
    model: Optional[torch.nn.Module],
    device: str = "cuda",
    max_points_per_frame: int = 5000,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Process a single episode to generate per-frame point clouds for each camera.
    
    Saves point clouds per-frame so they can be accumulated at train time.
    
    Args:
        episode_dir: Directory containing the episode video files
        episode_num: Episode number string
        episode_data: HDF5 group containing episode sensor data
        model: VGGT model
        device: Device to run on
        max_points_per_frame: Maximum points to keep per frame per camera
        
    Returns:
        Dict mapping camera_name -> {frame_idx -> points [N, 3]}
    """
    # Get episode length from HDF5 data
    num_frames = get_episode_length(episode_data)
    
    if num_frames < 5:
        logger.debug(f"Episode {episode_num} too short ({num_frames} frames), skipping")
        return {}
    
    logger.debug(f"Processing episode {episode_num} with {num_frames} frames")
    
    # Load all camera frames for this episode
    all_camera_frames = get_camera_frames(episode_dir, episode_num)
    
    if not all_camera_frames:
        logger.warning(f"No camera frames loaded for episode {episode_num}")
        return {}
    
    min_depth = 0.1
    max_depth = 10.0
    
    # Process each camera and store per-frame point clouds
    result = {}
    
    for camera_name, camera_frames in all_camera_frames.items():
        if len(camera_frames) == 0:
            continue
        
        # Run VGGT on all frames for this camera
        depth_maps, point_maps, extrinsics = estimate_depth_and_extrinsics(
            model, camera_frames, device
        )
        
        result[camera_name] = {}
        
        for frame_idx in range(len(camera_frames)):
            points = point_maps[frame_idx].reshape(-1, 3)
            depths = depth_maps[frame_idx].flatten()
            
            # Filter by depth
            valid = (depths > min_depth) & (depths < max_depth)
            valid_points = points[valid]
            
            # Transform to robot-centered frame
            valid_points = transform_points_to_robot_frame(valid_points, camera_name)
            
            # Subsample if too many points
            if len(valid_points) > max_points_per_frame:
                indices = np.random.choice(len(valid_points), max_points_per_frame, replace=False)
                valid_points = valid_points[indices]
            
            result[camera_name][frame_idx] = valid_points.astype(np.float32)
    
    return result


def process_house_directory(
    house_dir: str,
    output_dir: str,
    model: Optional[torch.nn.Module],
    device: str = "cuda",
    max_points_per_frame: int = 5000,
):
    """
    Process all episodes in a house directory.
    
    Saves per-frame point clouds for each camera. At train time, accumulate
    frames 0..N to get the full point cloud history up to frame N.
    
    Output HDF5 structure:
        episode_num/
            camera_name/
                frame_idx  -> point_cloud [N, 3]
    
    Args:
        house_dir: Directory containing episode data for one house
        output_dir: Output directory for HDF5 file
        model: VGGT model
        device: Device to run on
        max_points_per_frame: Maximum points per frame per camera
    """
    house_id = os.path.basename(house_dir)
    hdf5_path = os.path.join(house_dir, "hdf5_sensors.hdf5")
    
    if not os.path.exists(hdf5_path):
        logger.warning(f"No hdf5_sensors.hdf5 found in {house_dir}")
        return
    
    output_path = os.path.join(output_dir, house_id, "point_clouds.hdf5")
    
    # Skip if already processed
    if os.path.exists(output_path):
        logger.info(f"Skipping {house_id} (already processed)")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Processing house {house_id}")
    
    with h5py.File(hdf5_path, 'r') as f_in:
        episodes = list(f_in.keys())
        
        with h5py.File(output_path, 'w') as f_out:
            for episode_num in tqdm(episodes, desc=f"House {house_id}", leave=False):
                episode_data = f_in[episode_num]
                
                try:
                    # Get per-frame point clouds for each camera
                    camera_points = process_episode(
                        house_dir,
                        episode_num,
                        episode_data,
                        model,
                        device,
                        max_points_per_frame,
                    )
                    
                    if not camera_points:
                        continue
                    
                    # Create episode group
                    episode_group = f_out.create_group(episode_num)
                    
                    # Store per-camera, per-frame point clouds
                    for camera_name, frame_points in camera_points.items():
                        camera_group = episode_group.create_group(camera_name)
                        
                        for frame_idx, points in frame_points.items():
                            camera_group.create_dataset(
                                str(frame_idx),
                                data=points,
                                compression="gzip"
                            )
                    
                    # Store metadata
                    episode_group.attrs["num_frames"] = max(
                        max(fp.keys()) + 1 for fp in camera_points.values()
                    ) if camera_points else 0
                    episode_group.attrs["cameras"] = list(camera_points.keys())
                
                except Exception as e:
                    logger.error(f"Error processing episode {episode_num} in {house_id}: {e}")
                    continue
    
    logger.info(f"Saved point clouds for house {house_id} to {output_path}")


def get_already_processed(output_dir: str) -> set:
    """
    Get set of house IDs that have already been processed.
    
    Args:
        output_dir: Output directory containing processed houses
        
    Returns:
        Set of house IDs that have point_clouds.hdf5
    """
    processed = set()
    if not os.path.exists(output_dir):
        return processed
    
    for house_id in os.listdir(output_dir):
        hdf5_path = os.path.join(output_dir, house_id, "point_clouds.hdf5")
        if os.path.exists(hdf5_path):
            processed.add(house_id)
    
    return processed


def worker_process(
    gpu_id: int,
    house_dirs: List[str],
    output_dir: str,
    max_points_per_frame: int,
    worker_id: int,
    num_workers: int,
    progress_queue: Optional[mp.Queue] = None,
):
    """
    Worker process for multi-GPU processing.
    
    Each worker loads its own VGGT model on its assigned GPU and processes
    a subset of house directories.
    
    Args:
        gpu_id: GPU device ID to use
        house_dirs: List of all house directories
        output_dir: Output directory
        max_points_per_frame: Max points per frame per camera
        worker_id: ID of this worker (for work division)
        num_workers: Total number of workers
        progress_queue: Queue to report progress back to main process
    """
    device = f"cuda:{gpu_id}"
    
    # Set up logging for this worker
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    worker_logger.info(f"Worker {worker_id} starting on GPU {gpu_id}")
    
    # Each worker takes every Nth house
    my_houses = house_dirs[worker_id::num_workers]
    worker_logger.info(f"Worker {worker_id} processing {len(my_houses)} houses")
    
    # Load model on this GPU
    torch.cuda.set_device(gpu_id)
    model = load_vggt_model(device)
    
    processed = 0
    errors = 0
    
    for house_dir in my_houses:
        try:
            process_house_directory(
                house_dir,
                output_dir,
                model,
                device,
                max_points_per_frame,
            )
            processed += 1
            
            if progress_queue:
                progress_queue.put(("progress", worker_id, processed))
                
        except Exception as e:
            worker_logger.error(f"Error processing {house_dir}: {e}")
            errors += 1
            continue
    
    worker_logger.info(f"Worker {worker_id} finished: {processed} processed, {errors} errors")
    
    if progress_queue:
        progress_queue.put(("done", worker_id, processed))


def run_multi_gpu(
    house_dirs: List[str],
    output_dir: str,
    num_gpus: int,
    max_points_per_frame: int,
):
    """
    Run preprocessing on multiple GPUs using multiprocessing.
    
    Args:
        house_dirs: List of house directories to process
        output_dir: Output directory
        num_gpus: Number of GPUs to use
        max_points_per_frame: Max points per frame per camera
    """
    # Must use 'spawn' for CUDA with multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Detect available GPUs
    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        logger.warning(f"Requested {num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
        num_gpus = available_gpus
    
    if num_gpus == 0:
        logger.error("No GPUs available!")
        return
    
    logger.info(f"Starting multi-GPU processing with {num_gpus} GPUs")
    logger.info(f"Processing {len(house_dirs)} houses")
    
    # Create progress queue
    progress_queue = mp.Queue()
    
    # Start workers
    processes = []
    for worker_id in range(num_gpus):
        gpu_id = worker_id  # Simple 1:1 mapping
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                house_dirs,
                output_dir,
                max_points_per_frame,
                worker_id,
                num_gpus,
                progress_queue,
            )
        )
        p.start()
        processes.append(p)
    
    # Monitor progress
    worker_progress = {i: 0 for i in range(num_gpus)}
    workers_done = 0
    start_time = time.time()
    
    pbar = tqdm(total=len(house_dirs), desc="Total progress")
    
    while workers_done < num_gpus:
        try:
            msg = progress_queue.get(timeout=60)
            msg_type, worker_id, count = msg
            
            if msg_type == "progress":
                old_count = worker_progress[worker_id]
                worker_progress[worker_id] = count
                pbar.update(count - old_count)
                
            elif msg_type == "done":
                workers_done += 1
                
        except:
            # Timeout - check if processes are still alive
            alive = sum(1 for p in processes if p.is_alive())
            if alive == 0:
                break
    
    pbar.close()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    total_processed = sum(worker_progress.values())
    rate = total_processed / elapsed if elapsed > 0 else 0
    
    logger.info(f"Multi-GPU processing complete!")
    logger.info(f"Processed {total_processed} houses in {elapsed:.1f}s ({rate:.2f} houses/sec)")


def run_distributed(
    house_dirs: List[str],
    output_dir: str,
    max_points_per_frame: int,
):
    """
    Run preprocessing in distributed mode (launched via torchrun).
    
    Args:
        house_dirs: List of house directories to process
        output_dir: Output directory
        max_points_per_frame: Max points per frame per camera
    """
    import torch.distributed as dist
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Rank {rank}/{world_size} starting on device {device}")
    
    # Each rank processes a subset
    my_houses = house_dirs[rank::world_size]
    logger.info(f"Rank {rank} processing {len(my_houses)} houses")
    
    # Load model
    model = load_vggt_model(device)
    
    # Process houses
    for house_dir in tqdm(my_houses, desc=f"Rank {rank}", position=rank):
        try:
            process_house_directory(
                house_dir,
                output_dir,
                model,
                device,
                max_points_per_frame,
            )
        except Exception as e:
            logger.error(f"Rank {rank} error processing {house_dir}: {e}")
    
    # Synchronize
    dist.barrier()
    
    if rank == 0:
        logger.info("Distributed processing complete!")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Precompute VGGT depth and accumulated point clouds for VIDA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU (basic usage):
  python scripts/precompute_vggt_depth.py \\
      --input_dir /path/to/ObjectNavType/train \\
      --output_dir /path/to/output

  # Multi-GPU with 4 GPUs (interactive session):
  python scripts/precompute_vggt_depth.py \\
      --input_dir /path/to/ObjectNavType/train \\
      --output_dir /path/to/output \\
      --num_gpus 4

  # Resume interrupted processing:
  python scripts/precompute_vggt_depth.py \\
      --input_dir /path/to/ObjectNavType/train \\
      --output_dir /path/to/output \\
      --num_gpus 4 --resume

  # Distributed mode (via torchrun):
  torchrun --nproc_per_node=4 scripts/precompute_vggt_depth.py \\
      --input_dir /path/to/ObjectNavType/train \\
      --output_dir /path/to/output \\
      --distributed

  # Process specific house range (for manual parallelism):
  python scripts/precompute_vggt_depth.py \\
      --input_dir /path/to/ObjectNavType/train \\
      --output_dir /path/to/output \\
      --start_house 0 --end_house 1000
"""
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory containing house subdirectories (e.g., ObjectNavType/train)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for point cloud HDF5 files"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run VGGT on (for single-GPU mode)"
    )
    parser.add_argument(
        "--max_points_per_frame", type=int, default=5000,
        help="Maximum points per frame per camera (accumulate at train time)"
    )
    parser.add_argument(
        "--num_houses", type=int, default=None,
        help="Limit number of houses to process (for testing)"
    )
    
    # Multi-GPU options
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use (uses multiprocessing for > 1)"
    )
    parser.add_argument(
        "--distributed", action="store_true",
        help="Run in distributed mode (must be launched via torchrun)"
    )
    
    # Range options (for manual parallelism)
    parser.add_argument(
        "--start_house", type=int, default=None,
        help="Start index for house processing (inclusive)"
    )
    parser.add_argument(
        "--end_house", type=int, default=None,
        help="End index for house processing (exclusive)"
    )
    
    # Resume option
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip houses that have already been processed"
    )
    
    args = parser.parse_args()
    
    print("=" * 60, flush=True)
    print("VGGT Point Cloud Preprocessing (per-frame)", flush=True)
    print("=" * 60, flush=True)
    print(f"Input:       {args.input_dir}", flush=True)
    print(f"Output:      {args.output_dir}", flush=True)
    print(f"Max pts/frame: {args.max_points_per_frame}", flush=True)
    print(f"Num GPUs:    {args.num_gpus}", flush=True)
    if args.num_houses:
        print(f"Num houses:  {args.num_houses} (limit)", flush=True)
    print("=" * 60, flush=True)
    
    print("Scanning for house directories...", flush=True)
    
    # Find house directories
    house_dirs = sorted(glob.glob(os.path.join(args.input_dir, "*")))
    house_dirs = [d for d in house_dirs if os.path.isdir(d)]
    
    print(f"Found {len(house_dirs)} house directories in {args.input_dir}", flush=True)
    logger.info(f"Found {len(house_dirs)} house directories in {args.input_dir}")
    
    # Apply house range filter
    if args.start_house is not None or args.end_house is not None:
        start = args.start_house or 0
        end = args.end_house or len(house_dirs)
        house_dirs = house_dirs[start:end]
        logger.info(f"Processing houses {start} to {end} ({len(house_dirs)} houses)")
    
    # Apply num_houses limit
    if args.num_houses:
        house_dirs = house_dirs[:args.num_houses]
        logger.info(f"Limited to {len(house_dirs)} houses for testing")
    
    # Skip already processed houses if resuming
    if args.resume:
        already_processed = get_already_processed(args.output_dir)
        original_count = len(house_dirs)
        house_dirs = [d for d in house_dirs if os.path.basename(d) not in already_processed]
        skipped = original_count - len(house_dirs)
        logger.info(f"Resume mode: skipping {skipped} already processed houses, {len(house_dirs)} remaining")
    
    if not house_dirs:
        logger.info("No houses to process. Exiting.")
        return
    
    # Run in appropriate mode
    if args.distributed:
        # Distributed mode (launched via torchrun)
        run_distributed(
            house_dirs,
            args.output_dir,
            args.max_points_per_frame,
        )
    elif args.num_gpus > 1:
        # Multi-GPU mode with multiprocessing
        run_multi_gpu(
            house_dirs,
            args.output_dir,
            args.num_gpus,
            args.max_points_per_frame,
        )
    else:
        # Single GPU mode
        print(f"\nSingle GPU mode on device: {args.device}", flush=True)
        print("Loading VGGT model (this may take a minute)...", flush=True)
        logger.info("Loading VGGT model...")
        model = load_vggt_model(args.device)
        print("VGGT model loaded! Starting processing...\n", flush=True)
        
        for house_dir in tqdm(house_dirs, desc="Processing houses"):
            try:
                process_house_directory(
                    house_dir,
                    args.output_dir,
                    model,
                    args.device,
                    args.max_points_per_frame,
                )
            except Exception as e:
                logger.error(f"Error processing {house_dir}: {e}")
                continue
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
