"""
Pose extraction from MP4 video using MediaPipe.
Extracts BODY-25 style keypoints from video frames.
"""
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from pathlib import Path

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# MediaPipe Pose gives 33 landmarks.
# We map to BODY-25 format (25 keypoints) for compatibility.
# MediaPipe index -> BODY-25 index mapping
MP_TO_BODY25 = {
    0: 0,   # nose
    1: 1,   # neck (approximate: midpoint between ears)
    # Right arm
    12: 2,  # RShoulder
    14: 3,  # RElbow
    16: 4,  # RWrist
    # Left arm
    11: 5,  # LShoulder
    13: 6,  # LElbow
    15: 7,  # LWrist
    # MidHip (average of hips)
    # Right leg
    24: 9,  # RHip
    26: 10, # RKnee
    28: 11, # RAnkle
    # Left leg
    23: 12, # LHip
    25: 13, # LKnee
    27: 14, # LAnkle
    # Eyes
    8: 15,  # REye (left in MP)
    7: 16,  # LEye (right in MP)
    # Ears
    10: 17, # REar (left in MP)
    9: 18,  # LEar (right in MP)
    # Feet
    32: 19, # LBigToe (heel in MP)
    30: 22, # RBigToe (heel in MP)
}


def mediapipe_to_body25(mp_landmarks, h, w):
    """
    Convert MediaPipe 33 landmarks to BODY-25 format (25 keypoints).
    Each keypoint: [x, y, confidence]
    """
    body25 = np.zeros((25, 3), dtype=np.float32)

    # Fill mapped points
    for mp_idx, b25_idx in MP_TO_BODY25.items():
        lm = mp_landmarks[mp_idx]
        body25[b25_idx] = [lm.x, lm.y, lm.visibility]

    # Neck: midpoint of shoulders
    if 11 in [lm for lm in mp_landmarks] and 12 in [lm for lm in mp_landmarks]:
        lshoulder = mp_landmarks[11]
        rshoulder = mp_landmarks[12]
        neck_x = (lshoulder.x + rshoulder.x) / 2
        neck_y = (lshoulder.y + rshoulder.y) / 2
        neck_vis = min(lshoulder.visibility, rshoulder.visibility)
        body25[1] = [neck_x, neck_y, neck_vis]

    # Midhip: average of hips
    if 23 in [lm for lm in mp_landmarks] and 24 in [lm for lm in mp_landmarks]:
        lhip = mp_landmarks[23]
        rhip = mp_landmarks[24]
        body25[8] = [
            (lhip.x + rhip.x) / 2,
            (lhip.y + rhip.y) / 2,
            min(lhip.visibility, rhip.visibility),
        ]

    return body25


def extract_keypoints_from_mp4(video_path: str, max_frames: int = 300, fps_target: int = 15) -> np.ndarray:
    """
    Process MP4 video and extract BODY-25 keypoints.
    
    Returns:
        keypoints: np.ndarray of shape (num_frames, 25, 3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30

    # Calculate frame sampling to get fps_target
    sample_every = max(1, int(original_fps / fps_target))

    all_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames
        if frame_count % sample_every != 0:
            frame_count += 1
            continue

        frame_count += 1
        if len(all_frames) >= max_frames:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            body25 = mediapipe_to_body25(results.pose_landmarks.landmark, h, w)
            all_frames.append(body25)
        else:
            # No pose detected, add zeros
            all_frames.append(np.zeros((25, 3), dtype=np.float32))

    cap.release()

    if len(all_frames) == 0:
        raise ValueError("No frames extracted from video")

    return np.array(all_frames)
