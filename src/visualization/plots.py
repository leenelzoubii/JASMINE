"""
Pose visualization components.

Static matplotlib plots and interactive HTML/JS skeleton viewer.
Enhanced with multi-person bounding box visualization.
"""

import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.config import (
    BODY_25_KEYPOINTS,
    SKELETON_CONNECTIONS,
    SKELETON_COLOR_GROUPS,
    SKELETON_COLORS,
    JOINT_ANGLE_TRIPLETS,
)


# ============================================================
# MULTI-PERSON VISUALIZATION
# ============================================================

# Colors for different people
PERSON_COLORS_DICT = {
    'instructor': '#2ca02c',  # Green
    'child': '#d62728',        # Red
    'unknown': '#ff7f0e'      # Orange
}

PERSON_BG_COLORS = {
    'instructor': 'green',
    'child': 'red',
    'unknown': 'orange'
}


def plot_pose_skeleton_with_bounding_box(
    keypoints: np.ndarray, 
    ax: Optional[plt.Axes] = None,
    title: str = "Pose Skeleton",
    frame_idx: int = 0,
    person_label: str = "Person",
    show_bbox: bool = True,
    bbox_color: str = "#ff7f0e",
    bbox_linewidth: int = 2
) -> plt.Figure:
    """
    Plot a 2D pose skeleton with optional bounding box.
    
    Args:
        keypoints: numpy array of shape (frames, joints, 2+) or (joints, 2+)
        ax: Matplotlib axis (creates new figure if None)
        title: Plot title
        frame_idx: Frame index to visualize
        person_label: Label for the person (e.g., 'Instructor', 'Child')
        show_bbox: Whether to show bounding box
        bbox_color: Color for bounding box
        bbox_linewidth: Line width for bounding box
        
    Returns:
        fig: Matplotlib figure
    """
    if keypoints.ndim == 3:
        kp = keypoints[frame_idx]
    else:
        kp = keypoints
    
    coords = kp[:, :2]  # x, y only
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure
    
    # Calculate bounding box
    if show_bbox:
        valid_mask = (coords[:, 0] > 0) | (coords[:, 1] > 0)
        if np.any(valid_mask):
            valid_coords = coords[valid_mask]
            x_min = valid_coords[:, 0].min()
            x_max = valid_coords[:, 0].max()
            y_min = valid_coords[:, 1].min()
            y_max = valid_coords[:, 1].max()
            
            # Add small padding
            width = x_max - x_min
            height = y_max - y_min
            padding = 0.05
            x_min -= width * padding
            x_max += width * padding
            y_min -= height * padding
            y_max += height * padding
            
            # Draw bounding box
            bbox = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=bbox_linewidth,
                edgecolor=bbox_color,
                facecolor=bbox_color,
                alpha=0.15,
                label=f'{person_label} Bounding Box'
            )
            ax.add_patch(bbox)
    
    # Draw skeleton connections with color groups
    for group_name, connections in SKELETON_COLOR_GROUPS.items():
        color = SKELETON_COLORS[group_name]
        for j1, j2 in connections:
            if j1 < len(coords) and j2 < len(coords):
                if np.any(coords[j1] != 0) and np.any(coords[j2] != 0):
                    ax.plot(
                        [coords[j1, 0], coords[j2, 0]],
                        [coords[j1, 1], coords[j2, 1]],
                        color=color, linewidth=3, solid_capstyle='round',
                        zorder=2,
                    )
    
    # Draw joints with person-specific color
    for i in range(len(coords)):
        if np.any(coords[i] != 0):
            ax.scatter(coords[i, 0], coords[i, 1],
                      c=bbox_color, s=50, zorder=3, 
                      edgecolors='white', linewidth=1.5)
    
    # Add joint labels for key joints
    key_joints = [0, 1, 2, 5, 8, 9, 12]
    for i in key_joints:
        if i < len(coords) and np.any(coords[i] != 0):
            ax.annotate(
                f"{i}:{BODY_25_KEYPOINTS[i]}",
                (coords[i, 0], coords[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color='#333333',
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add person label as legend
    if show_bbox:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_multi_person_skeleton(
    keypoints_sequence: np.ndarray,
    person_info: List[Dict],
    ax: Optional[plt.Axes] = None,
    title: str = "Multi-Person Skeleton",
    frame_idx: int = 0
) -> plt.Figure:
    """
    Plot skeleton with multiple people, each with their own bounding box and label.
    
    Args:
        keypoints_sequence: numpy array of shape (frames, num_people, joints, 2+) 
                         OR list of keypoint arrays for different people
        person_info: List of person info dicts with 'classification' and 'bounding_box'
        ax: Matplotlib axis
        title: Plot title
        frame_idx: Frame index
        
    Returns:
        fig: Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure
    
    # Handle different input formats
    if isinstance(keypoints_sequence, list):
        # List of keypoint arrays (different people)
        all_keypoints = keypoints_sequence
    elif keypoints_sequence.ndim == 4:
        # (frames, people, joints, coords)
        if frame_idx < keypoints_sequence.shape[0]:
            all_keypoints = [keypoints_sequence[frame_idx, p] for p in range(keypoints_sequence.shape[1])]
        else:
            all_keypoints = [keypoints_sequence[0, 0]]
    else:
        # Single person fallback
        all_keypoints = [keypoints_sequence]
    
    # Track bounds for auto-scaling
    all_x = []
    all_y = []
    
    # Plot each person
    for person_idx, (kp, info) in enumerate(zip(all_keypoints, person_info)):
        if kp is None or (kp == 0).all():
            continue
            
        coords = kp[:, :2]
        
        # Get person classification
        classification = info.get('classification', 'unknown')
        
        # Get colors
        person_color = PERSON_COLORS_DICT.get(classification, '#ff7f0e')
        
        # Draw bounding box if available
        if 'bounding_box' in info:
            x_min, y_min, x_max, y_max = info['bounding_box']
            bbox = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=person_color,
                facecolor=person_color,
                alpha=0.1,
                label=f"{classification.capitalize()} ({person_idx + 1})"
            )
            ax.add_patch(bbox)
        
        # Draw skeleton connections
        for group_name, connections in SKELETON_COLOR_GROUPS.items():
            color = SKELETON_COLORS[group_name]
            for j1, j2 in connections:
                if j1 < len(coords) and j2 < len(coords):
                    if np.any(coords[j1] != 0) and np.any(coords[j2] != 0):
                        ax.plot(
                            [coords[j1, 0], coords[j2, 0]],
                            [coords[j1, 1], coords[j2, 1]],
                            color=color, linewidth=2.5, solid_capstyle='round',
                            zorder=2,
                        )
        
        # Draw joints
        for i in range(len(coords)):
            if np.any(coords[i] != 0):
                ax.scatter(coords[i, 0], coords[i, 1],
                          c=person_color, s=40, zorder=3,
                          edgecolors='white', linewidth=1)
        
        # Collect coordinates for bounds
        valid = coords[(coords[:, 0] > 0) | (coords[:, 1] > 0)]
        if len(valid) > 0:
            all_x.extend(valid[:, 0])
            all_y.extend(valid[:, 1])
    
    # Add labels for each person
    for person_idx, info in enumerate(person_info):
        if 'bounding_box' in info:
            x_min, y_min, x_max, y_max = info['bounding_box']
            classification = info.get('classification', 'unknown')
            label = PERSON_LABELS.get(classification, 'Person')
            
            # Add label text at bounding box
            ax.text(
                x_min, y_max + 0.02,
                f"{label} #{person_idx + 1}",
                fontsize=10,
                color=PERSON_COLORS_DICT.get(classification, '#333'),
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def create_interactive_multi_person_html(
    keypoints_sequence: np.ndarray,
    person_info: List[Dict],
    width: int = 600,
    height: int = 700
) -> str:
    """
    Create an interactive HTML skeleton viewer with multiple people and bounding boxes.
    
    Args:
        keypoints_sequence: numpy array of shape (frames, joints, 2+)
                           OR list where each index represents different person per frame
        person_info: List of person info dicts
        width: Canvas width
        height: Canvas height
        
    Returns:
        html: Complete HTML string with embedded JS/CSS
    """
    # Determine number of people and frames
    if isinstance(keypoints_sequence, list):
        num_people = len(keypoints_sequence[0]) if keypoints_sequence else 1
        num_frames = len(keypoints_sequence)
        coords_list = [kp[:, :, :2].tolist() for kp in keypoints_sequence]
    else:
        num_people = 1
        num_frames = keypoints_sequence.shape[0]
        coords_list = [keypoints_sequence[f, :, :2].tolist() for f in range(num_frames)]
    
    # Person colors for JavaScript
    person_colors = {}
    for info in person_info:
        classification = info.get('classification', 'unknown')
        person_colors[classification] = PERSON_COLORS_DICT.get(classification, '#ff7f0e')
    
    import json
    coords_json = json.dumps(coords_list)
    person_info_json = json.dumps(person_info)
    colors_json = json.dumps(person_colors)
    
    # Get person labels
    labels_list = []
    for info in person_info:
        classification = info.get('classification', 'unknown')
        labels_list.append(PERSON_LABELS.get(classification, 'Person'))
    labels_json = json.dumps(labels_list)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f8f9fa;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
            }}
            .container {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.1);
                padding: 20px;
                max-width: {width + 40}px;
            }}
            h2 {{
                text-align: center;
                color: #1a1a2e;
                margin-bottom: 16px;
                font-size: 18px;
            }}
            .legend {{
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-bottom: 12px;
                flex-wrap: wrap;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
            }}
            .legend-color {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
            }}
            canvas {{
                display: block;
                margin: 0 auto;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background: #fafafa;
                cursor: crosshair;
            }}
            .controls {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-top: 16px;
                padding: 12px;
                background: #f0f2f5;
                border-radius: 8px;
            }}
            .btn {{
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s;
            }}
            .btn-play {{
                background: #1f77b4;
                color: white;
            }}
            .btn-play:hover {{
                background: #1565a0;
            }}
            .btn-play.playing {{
                background: #d62728;
            }}
            .slider-container {{
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 4px;
            }}
            .slider-container label {{
                font-size: 12px;
                color: #666;
            }}
            input[type="range"] {{
                width: 100%;
                height: 6px;
                -webkit-appearance: none;
                background: #ddd;
                border-radius: 3px;
                outline: none;
            }}
            input[type="range"]::-webkit-slider-thumb {{
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                background: #1f77b4;
                border-radius: 50%;
                cursor: pointer;
            }}
            .info {{
                display: flex;
                justify-content: space-between;
                margin-top: 12px;
                font-size: 12px;
                color: #666;
            }}
            .person-label {{
                position: absolute;
                background: rgba(255,255,255,0.9);
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Multi-Person Pose Skeleton Viewer</h2>
            <div class="legend" id="legend"></div>
            <canvas id="skeleton" width="{width}" height="{height - 100}"></canvas>
            <div class="controls">
                <button class="btn btn-play" id="playBtn" onclick="togglePlay()">&#9654; Play</button>
                <div class="slider-container">
                    <label>Frame: <span id="frameNum">0</span> / {num_frames - 1}</label>
                    <input type="range" id="frameSlider" min="0" max="{num_frames - 1}" value="0"
                           oninput="goToFrame(parseInt(this.value))">
                </div>
            </div>
            <div class="info">
                <span>Total Frames: {num_frames}</span>
                <span>People Detected: {num_people}</span>
            </div>
        </div>

        <script>
            const coords = {coords_json};
            const personInfo = {person_info_json};
            const personColors = {colors_json};
            const personLabels = {labels_json};
            const numFrames = coords.length;
            const numPeople = personInfo.length;

            const canvas = document.getElementById('skeleton');
            const ctx = canvas.getContext('2d');

            // Create legend
            const legend = document.getElementById('legend');
            const uniqueLabels = [...new Set(personLabels)];
            uniqueLabels.forEach(label => {{
                const color = personColors[label.toLowerCase()] || '#666';
                legend.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-color" style="background:${{color}}"></div>
                        <span>${{label}}</span>
                    </div>
                `;
            }});

            let currentFrame = 0;
            let isPlaying = false;
            let playInterval = null;
            let fps = 10;

            // Compute bounds for normalization
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (let f = 0; f < numFrames; f++) {{
                for (let p = 0; p < coords[f].length; p++) {{
                    for (let j = 0; j < coords[f][p].length; j++) {{
                        const [x, y] = coords[f][p][j];
                        if (x !== 0 || y !== 0) {{
                            minX = Math.min(minX, x);
                            maxX = Math.max(maxX, x);
                            minY = Math.min(minY, y);
                            maxY = Math.max(maxY, y);
                        }}
                    }}
                }}
            }}

            const padding = 40;
            const scaleX = (canvas.width - 2 * padding) / (maxX - minX || 1);
            const scaleY = (canvas.height - 2 * padding) / (maxY - minY || 1);
            const scale = Math.min(scaleX, scaleY);

            function toCanvas(x, y) {{
                return [
                    padding + (x - minX) * scale,
                    padding + (y - minY) * scale
                ];
            }}

            function drawSkeleton(frameIdx) {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw each person
                for (let p = 0; p < coords[frameIdx].length; p++) {{
                    const personCoords = coords[frameIdx][p];
                    const personLabel = personLabels[p] || 'Person';
                    const personColor = personColors[personLabel.toLowerCase()] || '#666';
                    const bbox = personInfo[p]?.bounding_box;

                    // Draw bounding box
                    if (bbox) {{
                        const [x_min, y_min, x_max, y_max] = bbox;
                        const [cx_min, cy_min] = toCanvas(x_min, y_min);
                        const [cx_max, cy_max] = toCanvas(x_max, y_max);
                        
                        ctx.strokeStyle = personColor;
                        ctx.lineWidth = 2;
                        ctx.strokeRect(cx_min, cy_min, cx_max - cx_min, cy_max - cy_min);
                        
                        ctx.fillStyle = personColor + '20';
                        ctx.fillRect(cx_min, cy_min, cx_max - cx_min, cy_max - cy_min);
                        
                        // Draw label
                        ctx.fillStyle = personColor;
                        ctx.font = 'bold 12px sans-serif';
                        ctx.fillText(personLabel, cx_min, cy_min - 5);
                    }}

                    // Skeleton connections (same as single person)
                    const connections = [
                        [0, 1], [1, 2], [2, 3], [3, 4],
                        [1, 5], [5, 6], [6, 7],
                        [1, 8], [8, 9], [9, 10], [10, 11],
                        [8, 12], [12, 13], [13, 14],
                        [0, 15], [15, 17], [0, 16], [16, 18]
                    ];

                    connections.forEach(([j1, j2]) => {{
                        const [x1, y1] = personCoords[j1];
                        const [x2, y2] = personCoords[j2];

                        if ((x1 === 0 && y1 === 0) || (x2 === 0 && y2 === 0)) return;

                        const [cx1, cy1] = toCanvas(x1, y1);
                        const [cx2, cy2] = toCanvas(x2, y2);

                        ctx.beginPath();
                        ctx.moveTo(cx1, cy1);
                        ctx.lineTo(cx2, cy2);
                        ctx.strokeStyle = personColor;
                        ctx.lineWidth = 2;
                        ctx.lineCap = 'round';
                        ctx.stroke();
                    }});

                    // Draw joints
                    for (let j = 0; j < personCoords.length; j++) {{
                        const [x, y] = personCoords[j];
                        if (x === 0 && y === 0) continue;

                        const [cx, cy] = toCanvas(x, y);

                        ctx.beginPath();
                        ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
                        ctx.fillStyle = personColor;
                        ctx.fill();
                        ctx.strokeStyle = 'white';
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }}
                }}
            }}

            function goToFrame(idx) {{
                currentFrame = Math.max(0, Math.min(idx, numFrames - 1));
                document.getElementById('frameSlider').value = currentFrame;
                document.getElementById('frameNum').textContent = currentFrame;
                drawSkeleton(currentFrame);
            }}

            function togglePlay() {{
                const btn = document.getElementById('playBtn');
                if (isPlaying) {{
                    clearInterval(playInterval);
                    btn.innerHTML = '&#9654; Play';
                    btn.classList.remove('playing');
                    isPlaying = false;
                }} else {{
                    playInterval = setInterval(() => {{
                        goToFrame(currentFrame + 1);
                        if (currentFrame >= numFrames - 1) {{
                            clearInterval(playInterval);
                            btn.innerHTML = '&#9654; Play';
                            btn.classList.remove('playing');
                            isPlaying = false;
                        }}
                    }}, 1000 / fps);
                    btn.innerHTML = '&#9632;&#9632; Pause';
                    btn.classList.add('playing');
                    isPlaying = true;
                }}
            }}

            // Keyboard controls
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'ArrowLeft') goToFrame(currentFrame - 1);
                if (e.key === 'ArrowRight') goToFrame(currentFrame + 1);
                if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
            }});

            // Initial draw
            drawSkeleton(0);
        </script>
    </body>
    </html>
    """
    return html


# ============================================================
# LEGACY FUNCTIONS (Backward Compatibility)
# ============================================================

def plot_pose_skeleton(keypoints: np.ndarray, ax: Optional[plt.Axes] = None,
                       title: str = "Pose Skeleton",
                       frame_idx: int = 0) -> plt.Figure:
    """
    Plot a 2D pose skeleton for a single frame.

    Args:
        keypoints: numpy array of shape (frames, joints, 2+) or (joints, 2+)
        ax: Matplotlib axis (creates new figure if None)
        title: Plot title
        frame_idx: Frame index to visualize

    Returns:
        fig: Matplotlib figure
    """
    if keypoints.ndim == 3:
        kp = keypoints[frame_idx]
    else:
        kp = keypoints

    coords = kp[:, :2]  # x, y only

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    # Draw skeleton connections with color groups
    for group_name, connections in SKELETON_COLOR_GROUPS.items():
        color = SKELETON_COLORS[group_name]
        for j1, j2 in connections:
            if j1 < len(coords) and j2 < len(coords):
                # Only draw if both joints have valid positions
                if np.any(coords[j1] != 0) and np.any(coords[j2] != 0):
                    ax.plot(
                        [coords[j1, 0], coords[j2, 0]],
                        [coords[j1, 1], coords[j2, 1]],
                        color=color, linewidth=3, solid_capstyle='round',
                        zorder=2,
                    )

    # Draw joints
    for i in range(len(coords)):
        if np.any(coords[i] != 0):
            ax.scatter(coords[i, 0], coords[i, 1],
                      c='#1f77b4', s=50, zorder=3, edgecolors='white', linewidth=1.5)

    # Add joint labels for key joints
    key_joints = [0, 1, 2, 5, 8, 9, 12]  # Nose, Neck, Shoulders, MidHip, Hips
    for i in key_joints:
        if i < len(coords) and np.any(coords[i] != 0):
            ax.annotate(
                f"{i}:{BODY_25_KEYPOINTS[i]}",
                (coords[i, 0], coords[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color='#333333',
            )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_yaxis()  # Image coordinates
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_joint_angles_over_time(keypoints: np.ndarray, fps: int = 30,
                                 max_plots: int = 6) -> plt.Figure:
    """
    Plot key joint angles over time.

    Args:
        keypoints: numpy array of shape (frames, joints, 2)
        fps: Frames per second
        max_plots: Maximum number of angle plots to show

    Returns:
        fig: Matplotlib figure
    """
    from src.features.kinematic import compute_joint_angles

    angles = compute_joint_angles(keypoints)
    num_angles = angles.shape[1]
    num_plots = min(num_angles, max_plots)

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    time = np.arange(angles.shape[0]) / fps

    for i in range(num_plots):
        j1, j2, j3 = JOINT_ANGLE_TRIPLETS[i]
        j1_name = BODY_25_KEYPOINTS[j1] if j1 < len(BODY_25_KEYPOINTS) else f"J{j1}"
        j2_name = BODY_25_KEYPOINTS[j2] if j2 < len(BODY_25_KEYPOINTS) else f"J{j2}"
        j3_name = BODY_25_KEYPOINTS[j3] if j3 < len(BODY_25_KEYPOINTS) else f"J{j3}"

        axes[i].plot(time, np.degrees(angles[:, i]), linewidth=1.5, color='#1f77b4')
        axes[i].set_ylabel(f"{j1_name}-{j2_name}-{j3_name}\n(degrees)", fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 180)

    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle('Joint Angles Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_velocity_heatmap(keypoints: np.ndarray) -> plt.Figure:
    """
    Plot heatmap of joint velocities across frames.

    Args:
        keypoints: numpy array of shape (frames, joints, 2)

    Returns:
        fig: Matplotlib figure
    """
    from src.features.kinematic import compute_joint_velocities

    velocities = compute_joint_velocities(keypoints)

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        velocities.T,
        aspect='auto',
        cmap='YlOrRd',
        interpolation='nearest',
    )

    # Set y-ticks to joint names
    joint_labels = []
    for j in range(velocities.shape[1]):
        name = BODY_25_KEYPOINTS[j] if j < len(BODY_25_KEYPOINTS) else f"J{j}"
        joint_labels.append(name)

    ax.set_yticks(range(len(joint_labels)))
    ax.set_yticklabels(joint_labels, fontsize=8)
    ax.set_xlabel('Frame')
    ax.set_title('Joint Velocity Heatmap', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Velocity (px/frame)')

    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                            top_n: int = 20) -> plt.Figure:
    """
    Plot horizontal bar chart of top feature importances.

    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Number of top features to show

    Returns:
        fig: Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    ax.barh(range(len(top_names)), top_values, color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: List[str] = ['TD', 'ASD']) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class label names

    Returns:
        fig: Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=12)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f'{cm[i, j]}',
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=16, fontweight='bold',
            )

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    return fig


def create_interactive_skeleton_html(keypoints_sequence: np.ndarray,
                                      width: int = 600, height: int = 700) -> str:
    """
    Create an interactive HTML skeleton viewer with play/pause and frame slider.

    Args:
        keypoints_sequence: numpy array of shape (frames, joints, 2+)
        width: Canvas width in pixels
        height: Canvas height in pixels

    Returns:
        html: Complete HTML string with embedded JS/CSS
    """
    num_frames = keypoints_sequence.shape[0]
    coords = keypoints_sequence[:, :, :2].tolist()

    # Prepare skeleton connections data
    connections_data = []
    for group_name, connections in SKELETON_COLOR_GROUPS.items():
        color = SKELETON_COLORS[group_name]
        for j1, j2 in connections:
            connections_data.append({'from': j1, 'to': j2, 'color': color})

    # Convert to JSON-safe format
    import json
    coords_json = json.dumps(coords)
    connections_json = json.dumps(connections_data)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f8f9fa;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
            }}
            .container {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.1);
                padding: 20px;
                max-width: {width + 40}px;
            }}
            h2 {{
                text-align: center;
                color: #1a1a2e;
                margin-bottom: 16px;
                font-size: 18px;
            }}
            canvas {{
                display: block;
                margin: 0 auto;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background: #fafafa;
                cursor: crosshair;
            }}
            .controls {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-top: 16px;
                padding: 12px;
                background: #f0f2f5;
                border-radius: 8px;
            }}
            .btn {{
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s;
            }}
            .btn-play {{
                background: #1f77b4;
                color: white;
            }}
            .btn-play:hover {{
                background: #1565a0;
            }}
            .btn-play.playing {{
                background: #d62728;
            }}
            .slider-container {{
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 4px;
            }}
            .slider-container label {{
                font-size: 12px;
                color: #666;
            }}
            input[type="range"] {{
                width: 100%;
                height: 6px;
                -webkit-appearance: none;
                background: #ddd;
                border-radius: 3px;
                outline: none;
            }}
            input[type="range"]::-webkit-slider-thumb {{
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                background: #1f77b4;
                border-radius: 50%;
                cursor: pointer;
            }}
            .info {{
                display: flex;
                justify-content: space-between;
                margin-top: 12px;
                font-size: 12px;
                color: #666;
            }}
            .tooltip {{
                position: absolute;
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 6px 10px;
                border-radius: 4px;
                font-size: 11px;
                pointer-events: none;
                display: none;
                z-index: 100;
            }}
            .speed-control {{
                display: flex;
                align-items: center;
                gap: 4px;
            }}
            .speed-control label {{
                font-size: 12px;
                color: #666;
            }}
            .speed-control select {{
                padding: 4px 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Interactive Pose Skeleton Viewer</h2>
            <canvas id="skeleton" width="{width}" height="{height - 100}"></canvas>
            <div class="controls">
                <button class="btn btn-play" id="playBtn" onclick="togglePlay()">&#9654; Play</button>
                <div class="slider-container">
                    <label>Frame: <span id="frameNum">0</span> / {num_frames - 1}</label>
                    <input type="range" id="frameSlider" min="0" max="{num_frames - 1}" value="0"
                           oninput="goToFrame(parseInt(this.value))">
                </div>
                <div class="speed-control">
                    <label>Speed:</label>
                    <select id="speedSelect" onchange="changeSpeed(this.value)">
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="2">2x</option>
                        <option value="4">4x</option>
                    </select>
                </div>
            </div>
            <div class="info">
                <span>Total Frames: {num_frames}</span>
                <span id="hoverInfo">Hover over a joint for coordinates</span>
            </div>
        </div>
        <div class="tooltip" id="tooltip"></div>

        <script>
            const coords = {coords_json};
            const connections = {connections_json};
            const numFrames = coords.length;
            const numJoints = coords[0].length;

            const canvas = document.getElementById('skeleton');
            const ctx = canvas.getContext('2d');
            const tooltip = document.getElementById('tooltip');

            let currentFrame = 0;
            let isPlaying = false;
            let playInterval = null;
            let fps = 10;

            // Compute bounds for normalization
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (let f = 0; f < numFrames; f++) {{
                for (let j = 0; j < numJoints; j++) {{
                    const [x, y] = coords[f][j];
                    if (x !== 0 || y !== 0) {{
                        minX = Math.min(minX, x);
                        maxX = Math.max(maxX, x);
                        minY = Math.min(minY, y);
                        maxY = Math.max(maxY, y);
                    }}
                }}
            }}

            const padding = 40;
            const scaleX = (canvas.width - 2 * padding) / (maxX - minX || 1);
            const scaleY = (canvas.height - 2 * padding) / (maxY - minY || 1);
            const scale = Math.min(scaleX, scaleY);

            function toCanvas(x, y) {{
                return [
                    padding + (x - minX) * scale,
                    padding + (y - minY) * scale
                ];
            }}

            function drawSkeleton(frameIdx) {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw connections
                for (const conn of connections) {{
                    const [x1, y1] = coords[frameIdx][conn.from];
                    const [x2, y2] = coords[frameIdx][conn.to];

                    if ((x1 === 0 && y1 === 0) || (x2 === 0 && y2 === 0)) continue;

                    const [cx1, cy1] = toCanvas(x1, y1);
                    const [cx2, cy2] = toCanvas(x2, y2);

                    ctx.beginPath();
                    ctx.moveTo(cx1, cy1);
                    ctx.lineTo(cx2, cy2);
                    ctx.strokeStyle = conn.color;
                    ctx.lineWidth = 3;
                    ctx.lineCap = 'round';
                    ctx.stroke();
                }}

                // Draw joints
                for (let j = 0; j < numJoints; j++) {{
                    const [x, y] = coords[frameIdx][j];
                    if (x === 0 && y === 0) continue;

                    const [cx, cy] = toCanvas(x, y);

                    ctx.beginPath();
                    ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
                    ctx.fillStyle = '#1f77b4';
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }}
            }}

            function goToFrame(idx) {{
                currentFrame = Math.max(0, Math.min(idx, numFrames - 1));
                document.getElementById('frameSlider').value = currentFrame;
                document.getElementById('frameNum').textContent = currentFrame;
                drawSkeleton(currentFrame);
            }}

            function togglePlay() {{
                const btn = document.getElementById('playBtn');
                if (isPlaying) {{
                    clearInterval(playInterval);
                    btn.innerHTML = '&#9654; Play';
                    btn.classList.remove('playing');
                    isPlaying = false;
                }} else {{
                    playInterval = setInterval(() => {{
                        goToFrame(currentFrame + 1);
                        if (currentFrame >= numFrames - 1) {{
                            clearInterval(playInterval);
                            btn.innerHTML = '&#9654; Play';
                            btn.classList.remove('playing');
                            isPlaying = false;
                        }}
                    }}, 1000 / fps);
                    btn.innerHTML = '&#9632;&#9632; Pause';
                    btn.classList.add('playing');
                    isPlaying = true;
                }}
            }}

            function changeSpeed(speed) {{
                fps = parseInt(speed) * 10;
                if (isPlaying) {{
                    togglePlay();
                    togglePlay();
                }}
            }}

            // Hover detection
            canvas.addEventListener('mousemove', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;

                let found = false;
                for (let j = 0; j < numJoints; j++) {{
                    const [x, y] = coords[currentFrame][j];
                    if (x === 0 && y === 0) continue;

                    const [cx, cy] = toCanvas(x, y);
                    const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);

                    if (dist < 12) {{
                        const jointNames = {json.dumps(BODY_25_KEYPOINTS)};
                        tooltip.style.display = 'block';
                        tooltip.style.left = (e.pageX + 10) + 'px';
                        tooltip.style.top = (e.pageY - 30) + 'px';
                        tooltip.textContent = `${{jointNames[j]}}: (${{x.toFixed(1)}}, ${{y.toFixed(1)}})`;
                        found = true;
                        break;
                    }}
                }}

                if (!found) {{
                    tooltip.style.display = 'none';
                }}
            }});

            canvas.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});

            // Keyboard controls
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'ArrowLeft') goToFrame(currentFrame - 1);
                if (e.key === 'ArrowRight') goToFrame(currentFrame + 1);
                if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
            }});

            // Initial draw
            drawSkeleton(0);
        </script>
    </body>
    </html>
    """
    return html


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64
