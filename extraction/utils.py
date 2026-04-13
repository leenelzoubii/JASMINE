"""
Helper functions for video processing and pose data handling.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    Get video properties.
    
    Returns:
        width, height, total_frames, fps
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    return width, height, total_frames, fps

def extract_frames(video_path: str, output_dir: str, 
                  frame_skip: int = 1) -> List[str]:
    """
    Extract frames from video at specified intervals.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        frame_skip: Extract every Nth frame (1 = all frames)
    
    Returns:
        List of extracted frame paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            frame_path = Path(output_dir) / "04d"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            
        frame_count += 1
    
    cap.release()
    return frame_paths

def load_openpose_keypoints(json_path: str) -> Optional[np.ndarray]:
    """
    Load keypoints from OpenPose JSON output.
    
    Args:
        json_path: Path to OpenPose JSON file
    
    Returns:
        keypoints: numpy array of shape (25, 3) or None if no person detected
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not data.get('people'):
            return None
            
        person = data['people'][0]  # Take first person
        keypoints_2d = person.get('pose_keypoints_2d', [])
        
        if len(keypoints_2d) == 0:
            return None
            
        # BODY_25 has 25 joints × 3 values (x, y, confidence)
        keypoints = np.array(keypoints_2d, dtype=np.float32).reshape(25, 3)
        return keypoints
        
    except (json.JSONDecodeError, KeyError, IndexError):
        return None

def save_sequence_to_csv(keypoints_sequence: np.ndarray, 
                        output_path: str, 
                        action_label: Optional[int] = None,
                        asd_label: Optional[int] = None) -> None:
    """
    Save keypoint sequence to MMASD-style CSV format.
    
    Args:
        keypoints_sequence: numpy array of shape (frames, 25, 3)
        output_path: Path to save CSV
        action_label: Optional action label (0-10)
        asd_label: Optional ASD label (0 or 1)
    """
    import pandas as pd
    
    # Joint names for BODY_25
    joint_names = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
        "left_eye", "right_ear", "left_ear", "background_18", "background_19",
        "background_20", "background_21", "background_22", "background_23", "background_24"
    ]
    
    # Create column names
    columns = []
    for joint_name in joint_names:
        columns.extend([f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"])
    
    if action_label is not None:
        columns.append("Action_Label")
    if asd_label is not None:
        columns.append("ASD_Label")
    
    # Prepare data
    data = []
    for frame_idx in range(keypoints_sequence.shape[0]):
        frame_data = keypoints_sequence[frame_idx].flatten().tolist()
        if action_label is not None:
            frame_data.append(action_label)
        if asd_label is not None:
            frame_data.append(asd_label)
        data.append(frame_data)
    
    # Save to CSV
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)