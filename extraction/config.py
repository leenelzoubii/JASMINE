"""
OpenPose configuration settings.
"""

import os

# OpenPose paths (update these with your OpenPose installation)
OPENPOSE_DIR = r"C:\path\to\openpose"  # Windows
# OPENPOSE_DIR = "/path/to/openpose"   # Linux/Mac

# Model paths
OPENPOSE_BIN = os.path.join(OPENPOSE_DIR, "bin", "OpenPoseDemo.exe")  # Windows
# OPENPOSE_BIN = os.path.join(OPENPOSE_DIR, "build", "examples", "openpose", "openpose.bin")  # Linux

OPENPOSE_MODELS_DIR = os.path.join(OPENPOSE_DIR, "models")

# Processing parameters
BODY_25_MODEL = "BODY_25"
DEFAULT_FPS = 30
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

# Output settings
JSON_OUTPUT_DIR = "openpose_output"