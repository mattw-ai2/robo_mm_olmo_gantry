"""
VGGT Aligned Point Cloud Preprocessing

This version runs VGGT on ALL frames from ALL cameras together,
so the output world_points are in a consistent aligned coordinate frame.

Output structure (per house):
    point_clouds_aligned.hdf5/
        episode_num/
            camera_name/  (front, right, down, left)
                frame_idx -> point_cloud [N, 3] in ALIGNED world coordinates
            pose_enc -> [num_frames * 4 cameras, 9] camera pose encodings

Usage:
    python scripts/precompute_vggt_aligned.py \
        --input_dir /path/to/ObjectNavType/train \
        --output_dir /path/to/output \
        --start_house 0 --end_house 100 --resume
"""

import argparse
import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CAMERA_NAMES = ["front", "right", "down", "left"]


def load_vggt_model(device: str = "cuda"):
    """Load VGGT model."""
    try:
        from vggt.models.vggt import VGGT
        model = VGGT.from_pretrained("facebook/vggt-1b")
        model = model.to(device)
        model.eval()
        logger.info("Loaded VGGT model successfully")
        return model
    except ImportError as e:
        logger.error(f"VGGT not installed: {e}")
        return None


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> Optional[np.ndarray]:
    """Load all frames from a video file."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr) if max_frames is None else min(len(vr), max_frames)
        frames = vr.get_batch(list(range(num_frames))).asnumpy()
        return frames
    except:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        return np.stack(frames) if frames else None


def load_all_camera_frames(episode_dir: str, episode_num: str, max_frames: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load frames from all 4 cameras."""
    camera_frames = {}
    for camera in CAMERA_NAMES:
        video_path = os.path.join(episode_dir, f"raw_{camera}_camera__{episode_num}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(episode_dir, f"warped_{camera}_camera__{episode_num}.mp4")
        if os.path.exists(video_path):
            frames = load_video_frames(video_path, max_frames)
            if frames is not None:
                camera_frames[camera] = frames
    return camera_frames


def resize_to_vggt_size(images: np.ndarray, target_size: int = 518) -> np.ndarray:
    """Resize images to VGGT's expected input size (multiple of 14)."""
    N, H, W, C = images.shape
    if H == target_size and W == target_size:
        return images
    
    resized = np.zeros((N, target_size, target_size, C), dtype=images.dtype)
    for i in range(N):
        resized[i] = cv2.resize(images[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized


def process_episode_aligned(
    episode_dir: str,
    episode_num: str,
    model: torch.nn.Module,
    device: str = "cuda",
    max_points_per_frame: int = 5000,
    max_frames: Optional[int] = None,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Optional[np.ndarray]]:
    """
    Process episode with all cameras aligned.
    
    Returns:
        camera_points: Dict[camera_name, Dict[frame_idx, points]]
        pose_enc: [num_total_frames, 9] pose encodings for all frames
    """
    # Load all camera frames
    camera_frames = load_all_camera_frames(episode_dir, episode_num, max_frames)
    
    if not camera_frames or len(camera_frames) < 2:
        return {}, None
    
    # Find minimum frame count across cameras
    min_frames = min(len(f) for f in camera_frames.values())
    if min_frames < 2:
        return {}, None
    
    # Interleave frames: [front_0, right_0, down_0, left_0, front_1, right_1, ...]
    # This way VGGT sees all viewpoints at each timestep together
    all_frames = []
    frame_to_camera_idx = []  # Track which camera each frame came from
    
    for t in range(min_frames):
        for cam_idx, camera in enumerate(CAMERA_NAMES):
            if camera in camera_frames:
                all_frames.append(camera_frames[camera][t])
                frame_to_camera_idx.append((camera, t))
    
    all_frames = np.stack(all_frames)  # [num_total_frames, H, W, 3]
    logger.debug(f"Episode {episode_num}: {len(all_frames)} total frames from {len(camera_frames)} cameras")
    
    # Resize to VGGT size
    all_frames = resize_to_vggt_size(all_frames)
    
    # Run VGGT on all frames together
    with torch.no_grad():
        # Convert to tensor [1, S, C, H, W]
        frames_tensor = torch.from_numpy(all_frames).permute(0, 3, 1, 2).float() / 255.0
        frames_tensor = frames_tensor.unsqueeze(0).to(device)  # [1, S, C, H, W]
        
        outputs = model(frames_tensor)
        
        # world_points: [1, S, H, W, 3] - ALIGNED across all frames!
        world_points = outputs['world_points'].squeeze(0).cpu().numpy()  # [S, H, W, 3]
        pose_enc = outputs['pose_enc'].squeeze(0).cpu().numpy()  # [S, 9]
        depth_maps = outputs['depth'].squeeze(0).squeeze(-1).cpu().numpy()  # [S, H, W]
    
    # Split back into per-camera point clouds
    min_depth, max_depth = 0.1, 10.0
    result = {camera: {} for camera in CAMERA_NAMES if camera in camera_frames}
    
    for idx, (camera, frame_t) in enumerate(frame_to_camera_idx):
        points = world_points[idx].reshape(-1, 3)
        depths = depth_maps[idx].flatten()
        
        # Filter by depth
        valid = (depths > min_depth) & (depths < max_depth) & ~np.isnan(points).any(axis=1)
        valid_points = points[valid]
        
        # Subsample
        if len(valid_points) > max_points_per_frame:
            indices = np.random.choice(len(valid_points), max_points_per_frame, replace=False)
            valid_points = valid_points[indices]
        
        result[camera][frame_t] = valid_points.astype(np.float32)
    
    return result, pose_enc.astype(np.float32)


def process_house_directory(
    house_dir: str,
    output_dir: str,
    model: torch.nn.Module,
    device: str = "cuda",
    max_points_per_frame: int = 5000,
    max_frames: Optional[int] = None,
):
    """Process all episodes in a house."""
    house_id = os.path.basename(house_dir)
    hdf5_path = os.path.join(house_dir, "hdf5_sensors.hdf5")
    
    if not os.path.exists(hdf5_path):
        logger.warning(f"No hdf5_sensors.hdf5 in {house_dir}")
        return
    
    output_path = os.path.join(output_dir, house_id, "point_clouds_aligned.hdf5")
    
    if os.path.exists(output_path):
        logger.info(f"Skipping {house_id} (already processed)")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Processing house {house_id}")
    
    with h5py.File(hdf5_path, 'r') as f_in:
        episodes = list(f_in.keys())
        
        with h5py.File(output_path, 'w') as f_out:
            for episode_num in tqdm(episodes, desc=f"House {house_id}", leave=False):
                try:
                    camera_points, pose_enc = process_episode_aligned(
                        house_dir, episode_num, model, device,
                        max_points_per_frame, max_frames
                    )
                    
                    if not camera_points:
                        continue
                    
                    episode_group = f_out.create_group(episode_num)
                    
                    # Save per-camera, per-frame point clouds
                    for camera_name, frame_points in camera_points.items():
                        camera_group = episode_group.create_group(camera_name)
                        for frame_idx, points in frame_points.items():
                            camera_group.create_dataset(
                                str(frame_idx), data=points, compression="gzip"
                            )
                    
                    # Save pose encodings
                    if pose_enc is not None:
                        episode_group.create_dataset("pose_enc", data=pose_enc, compression="gzip")
                    
                    # Metadata
                    episode_group.attrs["num_frames"] = max(
                        max(fp.keys()) + 1 for fp in camera_points.values()
                    ) if camera_points else 0
                    episode_group.attrs["cameras"] = list(camera_points.keys())
                    episode_group.attrs["aligned"] = True
                    
                except Exception as e:
                    logger.error(f"Error processing episode {episode_num} in {house_id}: {e}")
                    continue
    
    logger.info(f"Saved aligned point clouds for house {house_id}")


def main():
    parser = argparse.ArgumentParser(description="Precompute ALIGNED point clouds using VGGT")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_points_per_frame", type=int, default=5000)
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames per camera (for testing)")
    parser.add_argument("--start_house", type=int, default=None)
    parser.add_argument("--end_house", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("VGGT ALIGNED Point Cloud Preprocessing")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Find house directories
    house_dirs = sorted(glob.glob(os.path.join(args.input_dir, "*")))
    house_dirs = [d for d in house_dirs if os.path.isdir(d)]
    
    if args.start_house is not None or args.end_house is not None:
        start = args.start_house or 0
        end = args.end_house or len(house_dirs)
        house_dirs = house_dirs[start:end]
    
    if args.resume:
        already_processed = set()
        if os.path.exists(args.output_dir):
            for house_id in os.listdir(args.output_dir):
                if os.path.exists(os.path.join(args.output_dir, house_id, "point_clouds_aligned.hdf5")):
                    already_processed.add(house_id)
        house_dirs = [d for d in house_dirs if os.path.basename(d) not in already_processed]
        logger.info(f"Resume: {len(already_processed)} already processed, {len(house_dirs)} remaining")
    
    logger.info(f"Processing {len(house_dirs)} houses")
    
    # Load model
    model = load_vggt_model(args.device)
    if model is None:
        return
    
    # Process houses
    for house_dir in tqdm(house_dirs, desc="Processing houses"):
        try:
            process_house_directory(
                house_dir, args.output_dir, model, args.device,
                args.max_points_per_frame, args.max_frames
            )
        except Exception as e:
            logger.error(f"Error processing {house_dir}: {e}")
            continue
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

