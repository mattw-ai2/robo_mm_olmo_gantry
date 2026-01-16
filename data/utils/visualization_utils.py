import os
from typing import Dict, List, Any, Optional, Tuple, Union
import io
import numpy as np
import torchvision.io
from PIL import Image, ImageDraw, ImageFont, ImageColor
from splitstream import splitfile
import xml.etree.ElementTree as ET


def get_camera_frame(
    base_dir: str, episode_num: str, frame_idx: int, camera_name: str = "front"
) -> np.ndarray:
    """Load a specific frame from a camera video."""
    # Try both raw and warped camera paths
    for prefix in ["raw", "warped"]:
        video_path = os.path.join(
            base_dir, f"{prefix}_{camera_name}_camera__{episode_num}.mp4"
        )
        if os.path.exists(video_path):
            # Read the video and extract the frame
            try:
                video = torchvision.io.read_video(video_path, pts_unit="sec")[0]
                if frame_idx < len(video):
                    return video[frame_idx].numpy()
            except Exception as e:
                print(f"Error reading video {video_path}: {e}")

    # If we get here, we couldn't find or read the video
    raise FileNotFoundError(
        f"Could not find or read video for {camera_name} camera, episode {episode_num}"
    )


def draw_text_on_image(
    text: str,
    width: int = 800,
    height: int = 600,
    font_size: int = 12,
    bg_color=(255, 255, 255),
) -> np.ndarray:
    """Draw text on a blank image."""
    # Create a blank image with the specified background color
    img = np.ones((height, width, 3), dtype=np.uint8) * np.array(
        bg_color, dtype=np.uint8
    )

    # Convert to PIL Image for text drawing
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Use a default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Draw text
    text_color = (0, 0, 0)  # Black text
    x, y = 10, 10
    line_height = font_size + 4

    for line in text.split("\n"):
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_height

    # Convert back to numpy array
    return np.array(pil_img)


def compose_2x2_grid(frames: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compose a 2x2 grid from four camera frames.

    Args:
        frames: Dictionary with camera names as keys and frames as values
            Should contain 'front', 'right', 'left', 'down' keys

    Returns:
        Combined grid image
    """
    # Verify that all required frames are present
    required_cameras = ["front", "right", "left", "down"]
    missing_cameras = [cam for cam in required_cameras if cam not in frames]

    if missing_cameras:
        # Fill missing frames with black images of the same size as existing ones
        if frames:
            # Use the first available frame to determine size
            example_frame = next(iter(frames.values()))
            h, w = example_frame.shape[:2]
            black_frame = np.zeros((h, w, 3), dtype=np.uint8)

            for cam in missing_cameras:
                frames[cam] = black_frame
        else:
            # No frames at all, create placeholder frames
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for cam in required_cameras:
                frames[cam] = black_frame

    # Create the 2x2 grid
    top_row = np.concatenate([frames["front"], frames["right"]], axis=1)
    bottom_row = np.concatenate([frames["left"], frames["down"]], axis=1)
    grid = np.concatenate([top_row, bottom_row], axis=0)

    return grid


def draw_point_goal_on_frame(
    frame, goal_pose, already_normalized=False, color=(0, 255, 0), radius=8
):
    """Draw a point goal on a frame.

    Args:
        frame: Frame to draw on
        goal_pose: Goal pose (x, y)
        already_normalized: Whether the goal pose is already normalized
        color: Color of the point
        radius: Radius of the point

    Returns:
        Frame with the point goal drawn on it
    """
    if not already_normalized:
        x, y = goal_pose
    else:
        x, y = goal_pose
        height, width = frame.shape[0], frame.shape[1]
        x = int(x * width)
        y = int(y * height)

    frame_copy = frame.copy()
    
    # Convert to PIL image
    pil_img = Image.fromarray(frame_copy)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw filled circle
    draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=color)
    
    # Draw black outline
    draw.ellipse([(x-(radius+2), y-(radius+2)), (x+(radius+2), y+(radius+2))], 
                 outline=(0, 0, 0), width=2)
    
    # Convert back to numpy array
    return np.array(pil_img)


def get_color_by_distance(
    distance: float, max_distance: float = 5.0
) -> Tuple[int, int, int]:
    """
    Generate a color based on distance using a gradient.

    Args:
        distance: The distance value
        max_distance: Maximum distance for normalization

    Returns:
        RGB color tuple (0-255 scale)
    """
    # Normalize distance to 0-1 range, capped at max_distance
    normalized = min(distance / max_distance, 1.0)

    # Create a color gradient: green (close) -> yellow -> red (far)
    if normalized < 0.5:
        # Green to yellow
        g = 255
        r = int(255 * (normalized * 2))
        b = 0
    else:
        # Yellow to red
        r = 255
        g = int(255 * (1 - (normalized - 0.5) * 2))
        b = 0

    return (r, g, b)


def burn_points_onto_frame(
    frame: np.ndarray,
    points: List[List[Union[int, float]]],
    distances: Optional[List[float]] = None,
    point_radius: int = 5,
    max_distance: float = 5.0,
    default_color: Tuple[int, int, int] = (0, 255, 0),
    normalized: bool = True,
) -> np.ndarray:
    """
    Burn points onto a frame with color gradient based on distance.

    Args:
        frame: The input frame (numpy array)
        points: List of [x, y] coordinates
        distances: Optional list of distances for color gradient
        point_radius: Radius of points to draw
        max_distance: Maximum distance for color normalization
        default_color: Default color if distances not provided
        normalized: Whether the points are normalized (in range 0-1) or in pixel coordinates

    Returns:
        Frame with burned points
    """
    # Make a copy to avoid modifying the original
    result_frame = frame.copy()

    # Handle empty points list
    if not points or len(points) == 0:
        return result_frame

    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Convert to PIL Image
    pil_img = Image.fromarray(result_frame)
    draw = ImageDraw.Draw(pil_img)

    # Draw each point
    valid_points = 0
    for i, point in enumerate(points):
        # Extract x, y coordinates
        if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
            x, y = point[0], point[1]

            # Convert normalized coordinates to pixel coordinates if needed
            if normalized:
                x_px = int(x * w)
                y_px = int(y * h)
            else:
                x_px, y_px = int(x), int(y)
        else:
            continue  # Skip if point format is invalid

        # Ensure coordinates are within frame boundaries
        if not (0 <= x_px < w and 0 <= y_px < h):
            continue

        # Determine color based on distance
        if distances and i < len(distances):
            color = get_color_by_distance(distances[i], max_distance)
        else:
            color = default_color

        # Draw the point
        draw.ellipse(
            [(x_px-point_radius, y_px-point_radius), 
             (x_px+point_radius, y_px+point_radius)], 
            fill=color
        )
        valid_points += 1

    # Convert back to numpy array
    return np.array(pil_img)


def parse_vlm_output(xml_str: str, img_shape: tuple) -> np.ndarray:
    """Returns (N,2) array of points"""
    try:
        f = io.BytesIO(xml_str.encode("utf-8"))
        xml_str = next(splitfile(f, format="xml"))
        root = ET.fromstring(xml_str)
        if root.tag == "point":
            point = np.array(
                [
                    round(float(root.get("x")) / 100 * img_shape[1]),
                    round(float(root.get("y")) / 100 * img_shape[0]),
                ]
            )
            points = np.expand_dims(point, 0)
        elif root.tag == "points":
            points = []
            i = 1
            while f"x{i}" in root.attrib:
                points.append(
                    [
                        round(float(root.get(f"x{i}")) / 100 * img_shape[1]),
                        round(float(root.get(f"y{i}")) / 100 * img_shape[0]),
                    ]
                )
                i += 1
            points = np.array(points)
        return points
    except ET.ParseError as e:
        print("Invalid XML:\n", xml_str)
        raise e


def visualize_data_samples(
    episode_info,
    good_frames,
    prompts,
    labels,
    base_dir,
    episode_num,
    output_dir,
):
    """
    Visualize data samples from selected frames of an episode.

    Args:
        episode_info (dict): Dictionary containing episode information
        good_frames (list): List of selected frame indices
        prompts (list): List of prompts for each frame
        labels (list): List of labels for each frame
        base_dir (str): Base directory for video files
        episode_num (str): Episode number
        output_dir (str): Directory to save visualizations
    """
    # Make sure prompts and labels are lists
    if not isinstance(prompts, list):
        prompts = [prompts]
    if not isinstance(labels, list):
        labels = [labels]

    # Process each selected frame
    for i, frame_idx in enumerate(good_frames):
        print(f"\nProcessing frame {frame_idx} ({i+1}/{len(good_frames)})")

        # Get prompt and label for this frame
        prompt = prompts[i] if i < len(prompts) else "No prompt available"
        label = labels[i] if i < len(labels) else (False, None, None, None, None, None)

        # Unpack label
        is_done, label_camera, point_type, label_x, label_y, label_distance = label
        print(
            f"Label: is_done={is_done}, camera={label_camera}, type={point_type}, coords=({label_x}, {label_y}), distance={label_distance}"
        )

        # Get path and object data for the selected frame
        path_data = episode_info["path_points"].get(frame_idx, {})
        object_data = episode_info["object_points"].get(frame_idx, {})

        # Load frames from all cameras
        camera_frames = {}
        for camera_name in ["front", "right", "left", "down"]:
            try:
                camera_frames[camera_name] = get_camera_frame(
                    base_dir, episode_num, frame_idx, camera_name
                )
            except Exception as e:
                print(f"Could not load {camera_name} camera frame: {e}")
                # Create a placeholder black frame
                if camera_frames:
                    example_shape = next(iter(camera_frames.values())).shape
                    camera_frames[camera_name] = np.zeros(example_shape, dtype=np.uint8)
                else:
                    camera_frames[camera_name] = np.zeros((480, 640, 3), dtype=np.uint8)

        # Process future path points if available
        future_path_points = {}
        future_path_distances = {}
        if path_data:
            try:
                for camera in ["front", "right", "left", "down"]:
                    if camera in path_data:
                        points = path_data[camera]
                        if not isinstance(points, list) or len(points) == 0:
                            continue

                        # Extract points and distances
                        point_list = []
                        distance_list = []
                        for point in points:
                            if (
                                isinstance(point, dict)
                                and "x" in point
                                and "y" in point
                            ):
                                x = float(point["x"])
                                y = float(point["y"])
                                if 0 <= x <= 1 and 0 <= y <= 1:
                                    point_list.append((x, y))
                                    if "distance" in point:
                                        distance_list.append(float(point["distance"]))

                        if point_list:  # Only add if we have valid points
                            future_path_points[camera] = point_list
                            if distance_list:
                                future_path_distances[camera] = distance_list
            except Exception as e:
                print(f"Error parsing future path points: {e}")
                future_path_points = {"front": [], "right": [], "left": [], "down": []}
                future_path_distances = {
                    "front": [],
                    "right": [],
                    "left": [],
                    "down": [],
                }

        # Process future object points if available
        future_object_points = {}
        if object_data:
            try:
                for camera in ["front", "right", "left", "down"]:
                    if camera in object_data:
                        points = object_data[camera]
                        if not isinstance(points, list) or len(points) == 0:
                            continue

                        # Extract points
                        point_list = []
                        for point in points:
                            if (
                                isinstance(point, dict)
                                and "x" in point
                                and "y" in point
                            ):
                                x = float(point["x"])
                                y = float(point["y"])
                                if 0 <= x <= 1 and 0 <= y <= 1:
                                    point_list.append((x, y))

                        if point_list:  # Only add if we have valid points
                            future_object_points[camera] = point_list
            except Exception as e:
                print(f"Error parsing future object points: {e}")
                future_object_points = {
                    "front": [],
                    "right": [],
                    "left": [],
                    "down": [],
                }

        # Burn future points onto frames
        for camera in ["front", "right", "left", "down"]:
            # Burn path points with distance-based coloring
            if (
                camera in future_path_points
                and camera in camera_frames
                and future_path_points[camera]
            ):
                camera_frames[camera] = burn_points_onto_frame(
                    frame=camera_frames[camera],
                    points=future_path_points[camera],
                    distances=future_path_distances.get(camera, None),
                    point_radius=10,
                    max_distance=5.0,
                    normalized=True,
                )

            # Burn object points in a different color
            if (
                camera in future_object_points
                and camera in camera_frames
                and future_object_points[camera]
            ):
                camera_frames[camera] = burn_points_onto_frame(
                    frame=camera_frames[camera],
                    points=future_object_points[camera],
                    default_color=(255, 0, 255),  # Magenta for object points
                    point_radius=12,
                    normalized=True,
                )

        # Compose the 2x2 grid of camera frames
        grid_frame = compose_2x2_grid(camera_frames)

        # Add a label marker for the selected point on the grid (if applicable)
        if not is_done and label_camera and label_x is not None and label_y is not None:
            # Since coordinates are now normalized to the entire grid, we can draw directly on grid_frame
            h, w = grid_frame.shape[:2]
            px = int(label_x * w)
            py = int(label_y * h)

            # Convert to PIL for drawing
            pil_grid = Image.fromarray(grid_frame)
            draw = ImageDraw.Draw(pil_grid)
            
            # Draw a distinctive marker for the selected point
            color = (0, 255, 255) if point_type == "path" else (0, 255, 0)  # Yellow for path, green for object
            
            # Draw larger circle
            draw.ellipse([(px-15, py-15), (px+15, py+15)], outline=color, width=3)
            # Draw filled center
            draw.ellipse([(px-5, py-5), (px+5, py+5)], fill=color)

            # Add a label with the distance if available
            if point_type == "path" and label_distance is not None:
                label_text = f"{label_distance:.2f}m"
                # Use a font
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except IOError:
                    font = ImageFont.load_default()
                
                draw.text((px + 20, py), label_text, fill=color, font=font)
            
            # Convert back to numpy
            grid_frame = np.array(pil_grid)

        # Extract scene information
        scene_info_data = episode_info.get("scene_info", {}).get(frame_idx, {})

        # Prepare sidebar information
        sidebar_info = [
            f"Episode: {episode_num}",
            f"Frame: {frame_idx} / {episode_info['sequence_length']}",
            f"Task: {episode_info.get('task_description', 'Unknown')}",
            "",
            "--- PROMPT ---",
        ]

        # Add prompt with line breaks
        prompt_lines = []
        current_line = ""
        words = prompt.split()
        max_width = 45  # Approximate characters per line

        for word in words:
            if len(current_line) + len(word) + 1 <= max_width:
                current_line += (" " if current_line else "") + word
            else:
                prompt_lines.append(current_line)
                current_line = word
        if current_line:
            prompt_lines.append(current_line)

        sidebar_info.extend(prompt_lines)
        sidebar_info.extend(
            [
                "",
                "--- LABEL ---",
            ]
        )

        # Add label information
        if is_done:
            sidebar_info.append("DONE - Task completed")
        elif label_camera:
            label_info = f"Camera: {label_camera}, Type: {point_type}"
            if label_x is not None and label_y is not None:
                label_info += f", Grid Position: ({label_x:.2f}, {label_y:.2f})"
            if label_distance is not None:
                label_info += f", Distance: {label_distance:.2f}m"
            sidebar_info.append(label_info)
        else:
            sidebar_info.append("No valid label")

        # Add scene information if available
        if scene_info_data:
            sidebar_info.extend(
                [
                    "",
                    "--- SCENE INFO ---",
                ]
            )
            if "floor" in scene_info_data and "color_name" in scene_info_data["floor"]:
                sidebar_info.append(
                    f"Floor color: {scene_info_data['floor']['color_name']}"
                )
            if "wall" in scene_info_data and "color_name" in scene_info_data["wall"]:
                sidebar_info.append(
                    f"Wall color: {scene_info_data['wall']['color_name']}"
                )

        # Add point statistics
        path_points_count = sum(len(points) for points in future_path_points.values())
        object_points_count = sum(
            len(points) for points in future_object_points.values()
        )

        if path_points_count > 0 or object_points_count > 0:
            sidebar_info.extend(
                ["", "--- POINTS ---", f"Path points: {path_points_count} total"]
            )
            for camera in ["front", "right", "left", "down"]:
                if camera in future_path_points and future_path_points[camera]:
                    sidebar_info.append(
                        f"  {camera}: {len(future_path_points[camera])} points"
                    )

            sidebar_info.append(f"Object points: {object_points_count} total")
            for camera in ["front", "right", "left", "down"]:
                if camera in future_object_points and future_object_points[camera]:
                    sidebar_info.append(
                        f"  {camera}: {len(future_object_points[camera])} points"
                    )

        # Create the sidebar with all text information
        sidebar_width = 500  # Wider to accommodate more text
        grid_height = grid_frame.shape[0]
        info_text = "\n".join(sidebar_info)

        # Calculate font size based on content length
        font_size = 14
        if len(sidebar_info) > 20:
            font_size = 12
        if len(sidebar_info) > 30:
            font_size = 10

        sidebar_img = draw_text_on_image(
            info_text, width=sidebar_width, height=grid_height, font_size=font_size
        )

        # Combine the grid and sidebar
        combined_img = np.concatenate([grid_frame, sidebar_img], axis=1)

        house_id = int(base_dir.split("/")[-1])

        # Get task description and clean it for filename
        task_description = episode_info.get("task_description", "unknown_task")
        # Remove special characters that might cause issues in filenames
        task_description = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in task_description
        )
        task_description = task_description[:30]  # Limit length for filename

        # Save the visualization with more detailed filename
        output_file = os.path.join(
            output_dir,
            f"frame_viz_house{house_id}_ep{episode_num}_frame{frame_idx}_{task_description}.png",
        )
        
        # Save using PIL
        Image.fromarray(combined_img).save(output_file)
        print(f"Saved visualization to {output_file}")


def visualize_processed_samples(
    grid_frames, prompts, labels, output_dir, episode_nums=None
):
    """
    Visualize processed samples with pre-composed grid frames, prompts, and labels.

    Args:
        grid_frames (list): List of pre-composed 2x2 grid frames
        prompts (list): List of prompts corresponding to each grid frame
        labels (list): List of label strings (XML or "DONE") for each frame
        output_dir (str): Directory to save visualization outputs
        episode_nums (str, optional): Optional episode number for filename
    """
    os.makedirs(output_dir, exist_ok=True)

    # Process each grid frame with its prompt and label
    for i, (grid_frame, prompt, label) in enumerate(zip(grid_frames, prompts, labels)):
        print(f"\nProcessing sample {i+1}/{len(grid_frames)}")

        # Get frame dimensions
        h, w = grid_frame.shape[:2]

        # Parse the label XML if it's not "DONE"
        if label == "DONE":
            is_done = True
            point_x, point_y = None, None
        else:
            is_done = False
            try:
                # Parse XML to get point coordinates
                if "<point" in label:
                    # Extract x and y using parse_vlm_output
                    points = parse_vlm_output(label, (h, w))
                    if points is not None and len(points) > 0:
                        # Convert back to normalized coordinates (0-1)
                        point_x = float(points[0][0]) / w
                        point_y = float(points[0][1]) / h
                    else:
                        point_x, point_y = None, None
                else:
                    point_x, point_y = None, None
            except Exception as e:
                print(f"Error parsing label '{label}': {e}")
                point_x, point_y = None, None

        # Mark the point on the grid frame if it exists
        grid_with_point = grid_frame.copy()
        if not is_done and point_x is not None and point_y is not None:
            # Convert normalized coordinates to pixel coordinates
            px = int(point_x * w)
            py = int(point_y * h)

            # Convert to PIL for drawing
            pil_grid = Image.fromarray(grid_with_point)
            draw = ImageDraw.Draw(pil_grid)
            
            # Draw a distinctive marker for the point
            color = (0, 255, 0)  # Green
            
            # Larger circle
            draw.ellipse([(px-15, py-15), (px+15, py+15)], outline=color, width=3)
            # Filled center
            draw.ellipse([(px-5, py-5), (px+5, py+5)], fill=color)
            
            # Convert back to numpy
            grid_with_point = np.array(pil_grid)

        # Prepare sidebar information
        sidebar_info = []

        # Add episode number if available
        if episode_nums[i] is not None:
            sidebar_info.append(f"Episode: {episode_nums[i]}")
            sidebar_info.append(f"Sample: {i+1}/{len(grid_frames)}")

        sidebar_info.extend(["", "--- PROMPT ---"])

        # Add prompt with line breaks for readability
        prompt_lines = []
        current_line = ""
        words = prompt.split()
        max_width = 45  # Approximate characters per line

        for word in words:
            if len(current_line) + len(word) + 1 <= max_width:
                current_line += (" " if current_line else "") + word
            else:
                prompt_lines.append(current_line)
                current_line = word
        if current_line:
            prompt_lines.append(current_line)

        sidebar_info.extend(prompt_lines)
        sidebar_info.extend(["", "--- LABEL ---"])

        # Add label information
        if is_done:
            sidebar_info.append("DONE - Task completed")
        elif point_x is not None and point_y is not None:
            sidebar_info.append(
                f"Point coordinates (normalized): ({point_x:.2f}, {point_y:.2f})"
            )
            sidebar_info.append(
                f"Point coordinates (pixels): ({int(point_x * w)}, {int(point_y * h)})"
            )
        else:
            sidebar_info.append("No valid point in label")
            sidebar_info.append(f"Raw label: {label}")

        # Create the sidebar with all text information
        sidebar_width = 500  # Wide enough for text
        info_text = "\n".join(sidebar_info)

        # Calculate font size based on content length
        font_size = 14
        if len(sidebar_info) > 20:
            font_size = 12
        if len(sidebar_info) > 30:
            font_size = 10

        sidebar_img = draw_text_on_image(
            info_text, width=sidebar_width, height=h, font_size=font_size
        )

        # Combine the grid and sidebar
        combined_img = np.concatenate([grid_with_point, sidebar_img], axis=1)

        # Create filename
        if episode_nums[i]:
            base_filename = f"sample_{episode_nums[i]}_{i + 1}"
        else:
            base_filename = f"sample_{i+1}"

        # Save the visualization using PIL
        output_file = os.path.join(output_dir, f"{base_filename}.png")
        Image.fromarray(combined_img).save(output_file)
        print(f"Saved visualization to {output_file}")