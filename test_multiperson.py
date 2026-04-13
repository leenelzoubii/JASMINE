"""Test multi-person tracking functionality."""
import sys
sys.path.insert(0, 'C:/Users/HP/autism-screening-pose')

print('Testing multi-person tracking imports...')

# Test data loader (always needed)
from src.data.loader import (
    calculate_person_size,
    classify_person_by_size,
    calculate_bounding_box,
    PERSON_COLORS,
    PERSON_LABELS,
    CHILD_HEIGHT_THRESHOLD,
    load_all_people_from_openpose_json,
    extract_child_keypoints
)
print('OK: Data loader imports')

# Test config (always needed)
from src.config import BODY_25_KEYPOINTS, PERSON_LABELS
print('OK: Config imports')

# Visualization needs matplotlib - skip if not available
try:
    from src.visualization.plots import (
        plot_pose_skeleton_with_bounding_box,
        plot_pose_skeleton
    )
    print('OK: Visualization imports')
except ImportError:
    print('SKIP: Visualization (matplotlib not installed)')

print('')
print('Testing person tracking functions...')

import numpy as np

# Test 1: Tall person (instructor)
instructor = np.zeros((25, 3), dtype=np.float32)
instructor[0] = [0.5, 0.1, 0.9]   # nose
instructor[8] = [0.5, 0.5, 0.9]   # midhip
instructor[4] = [0.3, 0.4, 0.9]   # right wrist
instructor[7] = [0.7, 0.4, 0.9]   # left wrist

# Test 2: Short person (child)
child = np.zeros((25, 3), dtype=np.float32)
child[0] = [0.5, 0.3, 0.9]   # nose
child[8] = [0.5, 0.6, 0.9]   # midhip
child[4] = [0.4, 0.5, 0.9]   # right wrist
child[7] = [0.6, 0.5, 0.9]   # left wrist

# Calculate metrics
i_size = calculate_person_size(instructor)
c_size = calculate_person_size(child)
i_class = classify_person_by_size(instructor)
c_class = classify_person_by_size(child)
i_bbox = calculate_bounding_box(instructor)
c_bbox = calculate_bounding_box(child)

print('')
print(f'Test Results:')
print(f'  Instructor: height={i_size.get("height",0):.3f}, class={i_class}')
print(f'  Child:     height={c_size.get("height",0):.3f}, class={c_class}')
print(f'  Instructor bbox: x={i_bbox[0]:.2f}-{i_bbox[2]:.2f}, y={i_bbox[1]:.2f}-{i_bbox[3]:.2f}')
print(f'  Child bbox:     x={c_bbox[0]:.2f}-{c_bbox[2]:.2f}, y={c_bbox[1]:.2f}-{c_bbox[3]:.2f}')

# Test classification threshold
print(f'')
print(f'Threshold used: {CHILD_HEIGHT_THRESHOLD}')

# Verify
assert i_class == 'instructor', f'Expected instructor, got {i_class}'
assert c_class == 'child', f'Expected child, got {c_class}'

print('')
print('=' * 50)
print('SUCCESS: Multi-person tracking is working!')
print('=' * 50)