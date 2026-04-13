import json
import os
import pandas as pd
import numpy as np

def batch_process_all_folders(input_root, output_root):
    # Mapping for Action_Label based on folder prefix from your loader.py
    ACTION_MAPPING = {
        'as': 0, 'bs': 1, 'ce': 2, 'dr': 3, 'fg': 4,
    'mfs': 5, 'ms': 6, 'sq': 7, 'tw': 8, 'sac': 9, 'tr': 10
    }

    MMASD_JOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_heel", "right_heel", "left_foot", "right_foot",
    ]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Walk through the directory structure
    for root, dirs, files in os.walk(input_root):
        # Identify folders that contain JSON files
        json_files = sorted([f for f in files if f.endswith('.json')])
        
        if not json_files:
            continue

        # Get folder name
        folder_name = os.path.basename(root)
        print(f"Processing sequence: {folder_name} ({len(json_files)} frames)")

        all_frames = []
        for file in json_files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    if not data.get('people') or len(data['people']) == 0:
                        # 25 joints * 3 (x,y,z) = 75 zeros for empty frames
                        raw_kp = [0.0] * 75
                    else:
                        # Taking the first person detected
                        raw_kp = data['people'][0]['pose_keypoints_2d']
                        
                        # OpenPose usually provides 75 values. If for some reason 
                        # it provides more/less, we pad/trim to exactly 75.
                        if len(raw_kp) > 75:
                            raw_kp = raw_kp[:75]
                        elif len(raw_kp) < 75:
                            raw_kp.extend([0.0] * (75 - len(raw_kp)))
                            
                    all_frames.append(raw_kp)
            except (json.JSONDecodeError, IOError) as e:
                print(f"   Warning: Could not read {file}. Skipping frame.")
                continue

        if not all_frames:
            continue

        # Create DataFrame
        columns = []
        for name in MMASD_JOINT_NAMES:
            columns.extend([f'{name}_x', f'{name}_y', f'{name}_z'])
        
        df = pd.DataFrame(all_frames, columns=columns)

# AUTOMATIC ACTION LABELING
        # Extract prefix (e.g., 'as' from 'as_20583...')
        prefix = folder_name.split('_')[0].lower()
        df['Action_Label'] = ACTION_MAPPING.get(prefix, 0)
        
        # ASD LABEL from folder name ending:
        # _y = yes (ASD = 1)
        # _n = no (TD = 0)
        # _i = inconclusive (exclude or mark as 0)
        folder_lower = folder_name.lower()
        if folder_lower.endswith('_y'):
            df['ASD_Label'] = 1  # ASD
        elif folder_lower.endswith('_n'):
            df['ASD_Label'] = 0  # Typically Developing
        else:
            df['ASD_Label'] = 0  # Default for inconclusive (_i) or unknown

        # SAVE THE FILE
        output_path = os.path.join(output_root, f"{folder_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"   Done! Saved to {output_path}")


# ============================================================
# RELATIVE PATHS - Uses data/raw by default
# ============================================================

# Get project root (parent of extraction folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: Already exists at data/raw
INPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')

# Output: CSV files will be saved here
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'csv')

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    print(f"Input folder:  {INPUT_PATH}")
    print(f"Output folder: {OUTPUT_PATH}")
    print("-" * 40)
    
    # Check if input exists
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input folder not found!")
        print(f"Create this folder and add your JSON files:")
        print(f"  {INPUT_PATH}")
        print("")
        print("Folder structure should be:")
        print("  data/raw_json/")
        print("    ├── action_001/")
        print("    │   └── frame_0001.json")
        print("    ├── action_002/")
        print("    │   └── frame_0001.json")
        print("    └── ...")
    else:
        batch_process_all_folders(INPUT_PATH, OUTPUT_PATH)
        print("")
        print("=" * 40)
        print("CONVERSION COMPLETE!")
        print(f"CSV files saved to: {OUTPUT_PATH}")