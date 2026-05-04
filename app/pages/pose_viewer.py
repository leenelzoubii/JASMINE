import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BODY_25_KEYPOINTS, DEFAULT_FPS
from src.data.loader import (
    load_all_people_from_openpose_json,
    load_openpose_json,
    load_csv_sequence,
    calculate_person_size,
    calculate_bounding_box,
    PERSON_LABELS,
    PERSON_COLORS,
)
from src.visualization.plots import (
    plot_pose_skeleton,
    plot_pose_skeleton_with_bounding_box,
    plot_joint_angles_over_time,
    plot_velocity_heatmap,
    create_interactive_skeleton_html,
    create_interactive_multi_person_html,
)

ACCENT = "#4b9b79"


def pose_viewer_page():
    st.markdown(f'''
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: {ACCENT}; font-weight: 700; margin-bottom: 0.5rem;">Interactive Pose Viewer</h1>
        <p style="color: #888; font-size: 1.1rem;">Upload pose data to visualize skeleton and kinematic features</p>
    </div>
    ''', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload OpenPose JSON or MMASD CSV file",
        type=['json', 'csv'],
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            keypoints_sequence = None
            person_info = []

            if uploaded_file.name.endswith('.json'):
                try:
                    keypoints_list, person_info_list = load_all_people_from_openpose_json(tmp_path)

                    if len(keypoints_list) > 1:
                        st.success(f"Detected {len(keypoints_list)} people in frame!")

                        st.subheader("Detected People")
                        for info in person_info_list:
                            label = PERSON_LABELS.get(info['classification'], 'Unknown')
                            size = info.get('size_metrics', {})
                            st.write(f"**Person {info['person_id'] + 1}** ({label}): "
                                   f"Height={size.get('height', 0):.3f}, "
                                   f"Arm Span={size.get('arm_span', 0):.3f}")

                        keypoints_sequence = np.array(keypoints_list)[np.newaxis, ...]
                    else:
                        kp = keypoints_list[0] if keypoints_list else np.zeros((25, 3))
                        keypoints_sequence = kp[np.newaxis, ...]
                        person_info = [{
                            'person_id': 0,
                            'classification': 'unknown',
                            'size_metrics': calculate_person_size(kp),
                            'bounding_box': calculate_bounding_box(kp)
                        }]

                except ImportError:
                    kp = load_openpose_json(tmp_path)
                    keypoints_sequence = kp[np.newaxis, ...]

                st.info(f"Single frame: {kp.shape}")

            elif uploaded_file.name.endswith('.csv'):
                kp, action_label, asd_label = load_csv_sequence(tmp_path)
                keypoints_sequence = kp
                st.info(f"Sequence: {kp.shape[0]} frames, {kp.shape[1]} joints")

                person_info = [{
                    'person_id': 0,
                    'classification': 'unknown',
                    'size_metrics': calculate_person_size(kp),
                    'bounding_box': calculate_bounding_box(kp)
                }]

            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "Skeleton + BBox", "Multi-Person Interactive", "Joint Angles", "Velocity Heatmap"
            ])

            with viz_tab1:
                st.subheader("2D Pose Skeleton with Bounding Box")
                frame_slider = st.slider(
                    "Frame", 0, keypoints_sequence.shape[0] - 1, 0,
                    key="skeleton_frame"
                )

                if len(person_info) > 1:
                    selected_person = st.selectbox(
                        "Select Person to View",
                        options=range(len(person_info)),
                        format_func=lambda x: f"Person {x+1} ({person_info[x].get('classification', 'unknown')})"
                    )
                else:
                    selected_person = 0

                person_class = person_info[selected_person].get('classification', 'unknown') if person_info else 'unknown'
                bbox_color = PERSON_COLORS.get(person_class, '#ff7f0e')
                person_label = PERSON_LABELS.get(person_class, 'Person')

                try:
                    kp = keypoints_sequence[frame_slider] if keypoints_sequence.ndim == 3 else keypoints_sequence
                    fig = plot_pose_skeleton_with_bounding_box(
                        keypoints_sequence,
                        frame_idx=frame_slider,
                        person_label=person_label,
                        show_bbox=True,
                        bbox_color=bbox_color
                    )
                except ImportError:
                    fig = plot_pose_skeleton(keypoints_sequence, frame_idx=frame_slider)

                st.pyplot(fig)
                plt.close(fig)

            with viz_tab2:
                st.subheader("Multi-Person Interactive Skeleton Viewer")
                st.markdown(
                    "*Use mouse hover to see joint coordinates. "
                    "Arrow keys to navigate frames. Space to play/pause.*"
                )

                try:
                    if len(person_info) > 1:
                        html = create_interactive_multi_person_html(keypoints_sequence, person_info)
                    else:
                        html = create_interactive_skeleton_html(keypoints_sequence)
                except ImportError:
                    html = create_interactive_skeleton_html(keypoints_sequence)

                st.components.v1.html(html, height=750, scrolling=True)

                if person_info:
                    st.markdown("### Legend")
                    cols = st.columns(min(len(person_info), 3))
                    for idx, info in enumerate(person_info):
                        with cols[idx % 3]:
                            classification = info.get('classification', 'unknown')
                            label = PERSON_LABELS.get(classification, 'Unknown')
                            color = PERSON_COLORS.get(classification, '#666')
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:8px;">'
                                f'<div style="width:16px;height:16px;background:{color};border-radius:4px;"></div>'
                                f'<span><b>{label}</b> (Person {idx+1})</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

            with viz_tab3:
                st.subheader("Joint Angles Over Time")
                if keypoints_sequence.shape[0] > 1:
                    coords_2d = keypoints_sequence[:, :, :2]
                    fig = plot_joint_angles_over_time(coords_2d)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Joint angles require multiple frames. Upload a CSV sequence.")

            with viz_tab4:
                st.subheader("Joint Velocity Heatmap")
                if keypoints_sequence.shape[0] > 1:
                    coords_2d = keypoints_sequence[:, :, :2]
                    fig = plot_velocity_heatmap(coords_2d)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Velocity heatmap requires multiple frames. Upload a CSV sequence.")

            st.subheader("Raw Keypoint Data")
            st.write(f"Shape: {keypoints_sequence.shape}")

            if st.checkbox("Show raw coordinates"):
                frame_to_show = st.slider("Frame", 0, keypoints_sequence.shape[0] - 1, 0, key="raw_frame")
                frame_data = keypoints_sequence[frame_to_show]
                df = pd.DataFrame(frame_data, columns=['X', 'Y', 'Confidence'])
                df.index = [f"{i}: {name}" for i, name in enumerate(BODY_25_KEYPOINTS)]
                st.dataframe(df)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    pose_viewer_page()