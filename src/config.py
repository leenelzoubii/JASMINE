"""Configuration constants for the autism screening pipeline."""

# BODY_25 keypoint indices and names
BODY_25_KEYPOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel"
]

NUM_KEYPOINTS = 25
COORD_DIM = 3  # x, y, confidence

# BODY_25 skeleton connections (for visualization)
SKELETON_CONNECTIONS = [
    (0, 1),   # Nose-Neck
    (1, 2),   # Neck-RShoulder
    (2, 3),   # RShoulder-RElbow
    (3, 4),   # RElbow-RWrist
    (1, 5),   # Neck-LShoulder
    (5, 6),   # LShoulder-LElbow
    (6, 7),   # LElbow-LWrist
    (1, 8),   # Neck-MidHip
    (8, 9),   # MidHip-RHip
    (9, 10),  # RHip-RKnee
    (10, 11), # RKnee-RAnkle
    (8, 12),  # MidHip-LHip
    (12, 13), # LHip-LKnee
    (13, 14), # LKnee-LAnkle
    (0, 15),  # Nose-REye
    (15, 17), # REye-REar
    (0, 16),  # Nose-LEye
    (16, 18), # LEye-LEar
    (14, 19), # LAnkle-LBigToe
    (19, 20), # LBigToe-LSmallToe
    (20, 21), # LSmallToe-LHeel
    (11, 22), # RAnkle-RBigToe
    (22, 23), # RBigToe-RSmallToe
    (23, 24), # RSmallToe-RHeel
]

# Skeleton connection colors
SKELETON_COLORS = {
    "head": "#1f77b4",       # blue
    "torso": "#2ca02c",      # green
    "left_limb": "#d62728",  # red
    "right_limb": "#ff7f0e", # orange
}

# Which connections belong to which color group
SKELETON_COLOR_GROUPS = {
    "head": [(0, 15), (15, 17), (0, 16), (16, 18)],
    "torso": [(0, 1), (1, 8), (8, 9), (8, 12)],
    "left_limb": [(1, 5), (5, 6), (6, 7), (12, 13), (13, 14), (14, 19), (19, 20), (20, 21)],
    "right_limb": [(1, 2), (2, 3), (3, 4), (9, 10), (10, 11), (11, 22), (22, 23), (23, 24)],
}

# Joint angle triplets (joint1, joint2, joint3) where joint2 is the vertex
JOINT_ANGLE_TRIPLETS = [
    # Right arm
    (1, 2, 3),   # Neck-RShoulder-RElbow
    (2, 3, 4),   # RShoulder-RElbow-RWrist
    # Left arm
    (1, 5, 6),   # Neck-LShoulder-LElbow
    (5, 6, 7),   # LShoulder-LElbow-LWrist
    # Right leg
    (8, 9, 10),  # MidHip-RHip-RKnee
    (9, 10, 11), # RHip-RKnee-RAnkle
    # Left leg
    (8, 12, 13), # MidHip-LHip-LKnee
    (12, 13, 14),# LHip-LKnee-LAnkle
    # Torso
    (5, 1, 8),   # LShoulder-Neck-MidHip
    (2, 1, 8),   # RShoulder-Neck-MidHip
]

# Inter-joint distance pairs
JOINT_DISTANCE_PAIRS = [
    (2, 5),   # RShoulder-LShoulder (shoulder width)
    (9, 12),  # RHip-LHip (hip width)
    (2, 12),  # RShoulder-LHip (diagonal)
    (5, 9),   # LShoulder-RHip (diagonal)
    (4, 7),   # RWrist-LWrist (hand distance)
    (11, 14), # RAnkle-LAnkle (foot distance)
    (1, 8),   # Neck-MidHip (torso length)
    (0, 8),   # Nose-MidHip (head-to-hip)
]

# Default FPS for MMASD dataset
DEFAULT_FPS = 30

# Model types
ML_MODEL_TYPES = ["rf", "svm"]
DL_MODEL_TYPES = ["lstm", "transformer"]
ALL_MODEL_TYPES = ML_MODEL_TYPES + DL_MODEL_TYPES

# Risk thresholds (ensemble confidence)
RISK_THRESHOLDS = {
    "low": 0.3,
    "moderate": 0.6,
    # >= 0.6 = high risk
}

# Person labels for multi-person tracking
PERSON_LABELS = {
    'instructor': 'Instructor',
    'child': 'Child/Subject',
    'unknown': 'Person'
}
