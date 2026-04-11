"""
Main script for extracting 2D skeletons from videos using OpenPose.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from extraction.config import (
    OPENPOSE_BIN, OPENPOSE_MODELS_DIR, JSON_OUTPUT_DIR,
    BODY_25_MODEL, DEFAULT_WIDTH, DEFAULT_HEIGHT
)
from extraction.utils import (
    get_video_info, extract_frames, load_openpose_keypoints, save_sequence_to_csv
)

def run_openpose_on_video(video_path: str, 
                         output_dir: str = JSON_OUTPUT_DIR,
                         frame_skip: int = 1,
                         display: bool = False,
                         render_pose: bool = False) -> str:
    """
    Run OpenPose on a video file.
    
    Args:
        video_path: Path to input video
        output_dir: Directory for JSON outputs
        frame_skip: Process every Nth frame
        display: Show OpenPose display window
        render_pose: Render pose overlays
    
    Returns:
        Path to output directory containing JSON files
    """
    if not os.path.exists(OPENPOSE_BIN):
        raise FileNotFoundError(f"OpenPose binary not found at: {OPENPOSE_BIN}")
    
    if not os.path.exists(OPENPOSE_MODELS_DIR):
        raise FileNotFoundError(f"OpenPose models not found at: {OPENPOSE_MODELS_DIR}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build OpenPose command
    cmd = [
        OPENPOSE_BIN,
        "--video", video_path,
        "--model_pose", BODY_25_MODEL,
        "--model_folder", OPENPOSE_MODELS_DIR,
        "--write_json", str(output_path),
        "--frame_step", str(frame_skip),
        "--number_people_max", "1",  # Only track one person
    ]
    
    if display:
        cmd.append("--display")
    
    if render_pose:
        cmd.extend(["--write_images", str(output_path), "--write_images_format", "jpg"])
    
    # Run OpenPose
    print(f"Running OpenPose on: {video_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("OpenPose completed successfully")
        return str(output_path)
        
    except subprocess.CalledProcessError as e:
        print(f"OpenPose failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

def extract_skeleton_sequence(json_dir: str) -> np.ndarray:
    """
    Extract keypoint sequence from OpenPose JSON outputs.
    
    Args:
        json_dir: Directory containing OpenPose JSON files
    
    Returns:
        keypoints_sequence: numpy array of shape (frames, 25, 3)
    """
    json_files = sorted(Path(json_dir).glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    keypoints_list = []
    
    for json_file in json_files:
        keypoints = load_openpose_keypoints(str(json_file))
        if keypoints is not None:
            keypoints_list.append(keypoints)
        else:
            # Use zeros if no person detected
            keypoints_list.append(np.zeros((25, 3), dtype=np.float32))
    
    if not keypoints_list:
        raise ValueError("No valid keypoints found in any JSON files")
    
    return np.array(keypoints_list, dtype=np.float32)

def process_video(video_path: str, 
                 output_csv: str,
                 action_label: Optional[int] = None,
                 asd_label: Optional[int] = None,
                 frame_skip: int = 1,
                 display: bool = False) -> None:
    """
    Complete pipeline: video -> OpenPose -> CSV sequence.
    
    Args:
        video_path: Path to input video
        output_csv: Path for output CSV file
        action_label: Optional action label
        asd_label: Optional ASD label
        frame_skip: Process every Nth frame
        display: Show OpenPose display
    """
    print(f"Processing video: {video_path}")
    
    # Get video info
    width, height, total_frames, fps = get_video_info(video_path)
    print(f"Video info: {width}x{height}, {total_frames} frames, {fps} FPS")
    
    # Run OpenPose
    json_dir = run_openpose_on_video(
        video_path=video_path,
        frame_skip=frame_skip,
        display=display
    )
    
    # Extract sequence
    keypoints_sequence = extract_skeleton_sequence(json_dir)
    print(f"Extracted sequence: {keypoints_sequence.shape}")
    
    # Save to CSV
    save_sequence_to_csv(
        keypoints_sequence=keypoints_sequence,
        output_path=output_csv,
        action_label=action_label,
        asd_label=asd_label
    )
    
    print(f"Saved sequence to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Extract 2D skeletons from video using OpenPose")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_csv", help="Path for output CSV file")
    parser.add_argument("--action_label", type=int, choices=range(11), 
                       help="Action label (0-10)")
    parser.add_argument("--asd_label", type=int, choices=[0, 1],
                       help="ASD label (0 or 1)")
    parser.add_argument("--frame_skip", type=int, default=1,
                       help="Process every Nth frame")
    parser.add_argument("--display", action="store_true",
                       help="Show OpenPose display window")
    
    args = parser.parse_args()
    
    try:
        process_video(
            video_path=args.video_path,
            output_csv=args.output_csv,
            action_label=args.action_label,
            asd_label=args.asd_label,
            frame_skip=args.frame_skip,
            display=args.display
        )
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()