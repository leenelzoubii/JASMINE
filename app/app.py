"""
Autism Screening via Pose Estimation - Streamlit Application

A privacy-preserving demo for autism spectrum disorder screening
using 2D pose estimation keypoints.
"""

import os
import sys
import tempfile
import json
import base64
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import (
    load_all_models,
    get_ensemble_prediction,
    get_risk_level,
    format_prediction_result,
    generate_report,
    load_comparison_results,
)
from src.config import BODY_25_KEYPOINTS, DEFAULT_FPS
from src.visualization.plots import (
    plot_pose_skeleton,
    plot_joint_angles_over_time,
    plot_velocity_heatmap,
    plot_feature_importance,
    plot_confusion_matrix,
    create_interactive_skeleton_html,
    fig_to_base64,
)


# Page config
st.set_page_config(
    page_title="Autism Screening via Pose Estimation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 4px;
    }
    .risk-low { color: #2ca02c; font-weight: 700; }
    .risk-moderate { color: #ff7f0e; font-weight: 700; }
    .risk-high { color: #d62728; font-weight: 700; }
    .info-box {
        background: #e8f4fd;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
        margin: 12px 0;
    }
    .flow-step {
        display: inline-block;
        background: #1f77b4;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .flow-arrow {
        display: inline-block;
        color: #999;
        font-size: 1.2rem;
        margin: 0 4px;
    }
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Model Comparison", "Run Inference", "Pose Viewer"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This demo uses 2D pose estimation keypoints "
    "from the MMASD dataset to screen for autism spectrum disorder. "
    "**No raw video or images are stored** - only skeletal keypoints, "
    "preserving complete privacy."
)


# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":
    st.markdown('<div class="main-header">Autism Spectrum Disorder Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Privacy-Preserving Analysis via 2D Pose Estimation</div>', unsafe_allow_html=True)

    # Flow diagram
    st.markdown("### How It Works")
    flow_col = st.container()
    with flow_col:
        st.markdown(
            '<div style="text-align:center; padding: 20px;">'
            '<span class="flow-step">Video Input</span>'
            '<span class="flow-arrow">→</span>'
            '<span class="flow-step">OpenPose 2D</span>'
            '<span class="flow-arrow">→</span>'
            '<span class="flow-step">Keypoints (25 joints)</span>'
            '<span class="flow-arrow">→</span>'
            '<span class="flow-step">Feature Extraction</span>'
            '<span class="flow-arrow">→</span>'
            '<span class="flow-step">ML/DL Models</span>'
            '<span class="flow-arrow">→</span>'
            '<span class="flow-step">Prediction</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Key stats
    st.markdown("### Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">25</div>'
            '<div class="metric-label">BODY_25 Keypoints</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">4</div>'
            '<div class="metric-label">Models Compared</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">200+</div>'
            '<div class="metric-label">Extracted Features</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">100%</div>'
            '<div class="metric-label">Privacy Preserving</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Info sections
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Feature Types")
        st.markdown(
            '<div class="info-box">'
            '<b>Kinematic Features</b><br>'
            'Joint angles, velocities, inter-joint distances, '
            'and body symmetry metrics extracted from pose sequences.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-box">'
            '<b>Statistical Features</b><br>'
            'Per-joint coordinate statistics (mean, std, min, max, median, range) '
            'across all frames.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-box">'
            '<b>Temporal & Frequency Features</b><br>'
            'Frame-to-frame differences, autocorrelation, and FFT power spectrum '
            'analysis for movement pattern detection.'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### Models")
        st.markdown(
            '<div class="info-box">'
            '<b>Random Forest</b><br>'
            'Ensemble of decision trees with feature importance. '
            'Robust to overfitting, provides interpretable results.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-box">'
            '<b>SVM (Support Vector Machine)</b><br>'
            'Kernel-based classifier effective for high-dimensional feature spaces. '
            'With RBF and linear kernels.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-box">'
            '<b>LSTM (Long Short-Term Memory)</b><br>'
            'Recurrent neural network that captures temporal dependencies '
            'in pose sequences. Bidirectional with 2 layers.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-box">'
            '<b>Transformer</b><br>'
            'Self-attention based model for sequence classification. '
            'Captures long-range temporal patterns with positional encoding.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Privacy note
    st.markdown("---")
    st.warning(
        "**Privacy Note:** This system processes only 2D skeletal keypoints "
        "(x, y coordinates). No raw video frames, images, or personally identifiable "
        "visual data are stored or transmitted."
    )


# ============================================================
# MODEL COMPARISON PAGE
# ============================================================
elif page == "Model Comparison":
    st.title("Model Comparison Dashboard")
    st.markdown("Side-by-side comparison of all trained models.")

    # Load comparison results
    results_path = PROJECT_ROOT / "models" / "comparison_results.json"
    results = load_comparison_results(str(results_path))

    if results is None:
        st.warning(
            "No comparison results found. Run the training pipeline first:\n"
            "```python\n"
            "python train.py --data_dir /path/to/mmasd\n"
            "```"
        )

        # Allow manual upload of results
        uploaded = st.file_uploader("Or upload comparison_results.json", type=['json'])
        if uploaded:
            results = json.load(uploaded)

    if results:
        # View toggle
        view_mode = st.radio(
            "View Mode:",
            ["All Models", "ML Only", "DL Only"],
            horizontal=True,
        )

        # Comparison table
        st.subheader("Performance Metrics")
        comparison_data = results.get('comparison', [])

        if comparison_data:
            import pandas as pd
            df = pd.DataFrame(comparison_data)

            if view_mode == "ML Only":
                df = df[df['Model'].isin(['RF', 'SVM'])]
            elif view_mode == "DL Only":
                df = df[df['Model'].isin(['LSTM', 'TRANSFORMER'])]

            # Style the dataframe
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Visual comparison chart
            st.subheader("Accuracy Comparison")
            chart_data = df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].astype(float)
            st.bar_chart(chart_data)

        # Confusion matrices
        st.subheader("Confusion Matrices")
        models_data = results.get('models', {})

        cols = st.columns(2)
        for i, (model_type, model_data) in enumerate(models_data.items()):
            if view_mode == "ML Only" and model_type in ['lstm', 'transformer']:
                continue
            if view_mode == "DL Only" and model_type in ['rf', 'svm']:
                continue

            cm = model_data.get('confusion_matrix', [])
            if cm:
                cm_array = np.array(cm)
                fig = plot_confusion_matrix(
                    np.array([0, 1]),  # placeholder
                    np.array([0, 1]),  # placeholder
                )
                # Override with actual data
                fig, ax = plt.subplots(figsize=(5, 4))
                import matplotlib.pyplot as plt
                im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
                ax.set_title(f'{model_type.upper()} Confusion Matrix', fontsize=14, fontweight='bold')
                labels = ['TD', 'ASD']
                ax.set_xticks([0, 1])
                ax.set_xticklabels(labels, fontsize=12)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(labels, fontsize=12)

                thresh = cm_array.max() / 2.0
                for ii in range(cm_array.shape[0]):
                    for jj in range(cm_array.shape[1]):
                        ax.text(jj, ii, f'{cm_array[ii, jj]}',
                               ha='center', va='center',
                               color='white' if cm_array[ii, jj] > thresh else 'black',
                               fontsize=16, fontweight='bold')

                ax.set_ylabel('True Label', fontsize=12)
                ax.set_xlabel('Predicted Label', fontsize=12)
                plt.tight_layout()

                with cols[i % 2]:
                    st.pyplot(fig)
                    plt.close(fig)

        # Feature importance
        st.subheader("Feature Importance (Random Forest)")
        model_data = models_data.get('rf', {})
        top_features = model_data.get('top_features', {})

        if top_features:
            feat_names = list(top_features.keys())
            feat_values = np.array(list(top_features.values()))
            fig = plot_feature_importance(feat_names, feat_values, top_n=15)
            st.pyplot(fig)
            plt.close(fig)


# ============================================================
# RUN INFERENCE PAGE
# ============================================================
elif page == "Run Inference":
    st.title("Run Inference")
    st.markdown("Upload pose data and get predictions from all models.")

    # Load models
    models_dir = PROJECT_ROOT / "models"
    models = load_all_models(str(models_dir))

    if not models:
        st.warning(
            "No trained models found. Train models first or upload them.\n"
            "Expected files: `rf_model.pkl`, `svm_model.pkl`, `lstm_model.pth`, `transformer_model.pth`"
        )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload OpenPose JSON or MMASD CSV file",
        type=['json', 'csv'],
    )

    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            if uploaded_file.name.endswith('.json'):
                # Load OpenPose JSON
                from src.data.loader import load_openpose_json
                kp = load_openpose_json(tmp_path)
                keypoints_sequence = kp[np.newaxis, ...]  # (1, 25, 3)
                st.info(f"Loaded OpenPose JSON: {kp.shape[0]} joints detected")

            elif uploaded_file.name.endswith('.csv'):
                # Load MMASD CSV
                from src.data.loader import load_csv_sequence
                kp, action_label, asd_label = load_csv_sequence(tmp_path)
                keypoints_sequence = kp
                st.info(
                    f"Loaded MMASD CSV: {kp.shape[0]} frames, {kp.shape[1]} joints. "
                    f"{'ASD Label: ' + str(asd_label) if asd_label is not None else ''}"
                )

            # Process features
            from src.features.kinematic import extract_kinematic_features
            from src.features.statistical import extract_all_features

            # For ML: extract flat features
            coords_2d = keypoints_sequence[:, :, :2]
            kinematic_feats, kinematic_names = extract_kinematic_features(coords_2d)
            stat_feats, stat_names = extract_all_features(coords_2d)

            all_features = np.concatenate([kinematic_feats, stat_feats])
            all_names = kinematic_names + stat_names

            # For DL: use raw sequence
            dl_sequence = keypoints_sequence.reshape(
                keypoints_sequence.shape[0], -1
            )  # (frames, joints*coords)

            # Get predictions
            if models:
                predictions = get_ensemble_prediction(models, all_features, dl_sequence)
                formatted = format_prediction_result(predictions)

                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")

                # Ensemble result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Ensemble ASD Probability",
                        f"{formatted['ensemble_probability']:.1%}",
                    )
                with col2:
                    risk_color = formatted['risk_color']
                    st.markdown(
                        f'<div style="background:{risk_color};color:white;padding:16px;'
                        f'border-radius:10px;text-align:center;font-size:1.3rem;'
                        f'font-weight:700;">{formatted["risk_level"]}</div>',
                        unsafe_allow_html=True,
                    )
                with col3:
                    n_models = len(predictions)
                    st.metric("Models Used", n_models)

                # Individual model predictions
                st.subheader("Individual Model Predictions")
                for model_name, model_data in formatted['model_predictions'].items():
                    col_a, col_b, col_c = st.columns([2, 2, 1])
                    with col_a:
                        st.markdown(f"**{model_name}**")
                    with col_b:
                        st.progress(model_data['probability'])
                        st.caption(f"{model_data['probability']:.1%}")
                    with col_c:
                        st.markdown(
                            f'<span style="color:{model_data["color"]};font-weight:600;">'
                            f'{model_data["risk_level"]}</span>',
                            unsafe_allow_html=True,
                        )

                # Generate report
                report = generate_report(uploaded_file.name, formatted)
                st.subheader("Prediction Report")
                st.text_area("Report", report, height=300)

                # Download button
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"prediction_report_{uploaded_file.name}.txt",
                    mime="text/plain",
                )
            else:
                st.info("Models not loaded. Showing feature extraction only.")

                # Show extracted features
                st.subheader("Extracted Features")
                st.write(f"Total features: {len(all_features)}")
                st.write(f"Top 20 features by magnitude:")
                top_indices = np.argsort(np.abs(all_features))[::-1][:20]
                for idx in top_indices:
                    st.write(f"  {all_names[idx]}: {all_features[idx]:.4f}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            os.unlink(tmp_path)


# ============================================================
# POSE VIEWER PAGE (Enhanced with Multi-Person)
# ============================================================
elif page == "Pose Viewer":
    st.title("Interactive Pose Viewer")
    st.markdown("Upload pose data to visualize the skeleton and kinematic features. **Multi-person tracking with bounding boxes included!**")

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
                # Try multi-person loading first
                try:
                    from src.data.loader import (
                        load_all_people_from_openpose_json,
                        calculate_person_size,
                        classify_person_by_size,
                        calculate_bounding_box,
                        PERSON_LABELS
                    )
                    
                    # Load multi-person data
                    keypoints_list, person_info_list = load_all_people_from_openpose_json(tmp_path)
                    
                    if len(keypoints_list) > 1:
                        st.success(f"Detected {len(keypoints_list)} people in frame!")
                        
                        # Show detected people info
                        st.subheader("Detected People")
                        for info in person_info_list:
                            label = PERSON_LABELS.get(info['classification'], 'Unknown')
                            size = info.get('size_metrics', {})
                            st.write(f"**Person {info['person_id'] + 1}** ({label}): "
                                   f"Height={size.get('height', 0):.3f}, "
                                   f"Arm Span={size.get('arm_span', 0):.3f}")
                        
                        # Create sequence for visualization
                        keypoints_sequence = np.array(keypoints_list)[np.newaxis, ...]
                    else:
                        # Single person fallback
                        kp = keypoints_list[0] if keypoints_list else np.zeros((25, 3))
                        keypoints_sequence = kp[np.newaxis, ...]
                        person_info = [{
                            'person_id': 0,
                            'classification': 'unknown',
                            'size_metrics': calculate_person_size(kp),
                            'bounding_box': calculate_bounding_box(kp)
                        }]
                        
                except ImportError:
                    # Fallback to single person
                    from src.data.loader import load_openpose_json
                    kp = load_openpose_json(tmp_path)
                    keypoints_sequence = kp[np.newaxis, ...]

                st.info(f"Single frame: {kp.shape}")

            elif uploaded_file.name.endswith('.csv'):
                from src.data.loader import load_csv_sequence
                kp, action_label, asd_label = load_csv_sequence(tmp_path)
                keypoints_sequence = kp
                st.info(f"Sequence: {kp.shape[0]} frames, {kp.shape[1]} joints")
                
                # Create basic person info for CSV
                from src.data.loader import calculate_person_size, calculate_bounding_box, PERSON_LABELS
                person_info = [{
                    'person_id': 0,
                    'classification': 'unknown',
                    'size_metrics': calculate_person_size(kp),
                    'bounding_box': calculate_bounding_box(kp)
                }]

            # Visualization tabs
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "Skeleton + BBox", "Multi-Person Interactive", "Joint Angles", "Velocity Heatmap"
            ])

            with viz_tab1:
                st.subheader("2D Pose Skeleton with Bounding Box")
                frame_slider = st.slider(
                    "Frame", 0, keypoints_sequence.shape[0] - 1, 0,
                    key="skeleton_frame"
                )
                
                # Show person selector
                if len(person_info) > 1:
                    selected_person = st.selectbox(
                        "Select Person to View",
                        options=range(len(person_info)),
                        format_func=lambda x: f"Person {x+1} ({person_info[x].get('classification', 'unknown')})"
                    )
                else:
                    selected_person = 0
                
                # Get appropriate color for this person
                person_class = person_info[selected_person].get('classification', 'unknown') if person_info else 'unknown'
                from src.data.loader import PERSON_COLORS
                bbox_color = PERSON_COLORS.get(person_class, '#ff7f0e')
                
                # Get title
                person_label = PERSON_LABELS.get(person_class, 'Person')
                
                # Use enhanced plotting with bounding box
                try:
                    from src.visualization.plots import plot_pose_skeleton_with_bounding_box
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
                
                # Use multi-person HTML viewer if available
                try:
                    from src.visualization.plots import create_interactive_multi_person_html
                    
                    if len(person_info) > 1:
                        html = create_interactive_multi_person_html(keypoints_sequence, person_info)
                    else:
                        html = create_interactive_skeleton_html(keypoints_sequence)
                except ImportError:
                    html = create_interactive_skeleton_html(keypoints_sequence)
                
                st.components.v1.html(html, height=750, scrolling=True)
                
                # Show legend
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

            # Raw data table
            st.subheader("Raw Keypoint Data")
            st.write(f"Shape: {keypoints_sequence.shape}")

            if st.checkbox("Show raw coordinates"):
                frame_to_show = st.slider("Frame", 0, keypoints_sequence.shape[0] - 1, 0, key="raw_frame")
                import pandas as pd
                frame_data = keypoints_sequence[frame_to_show]
                df = pd.DataFrame(frame_data, columns=['X', 'Y', 'Confidence'])
                df.index = [f"{i}: {name}" for i, name in enumerate(BODY_25_KEYPOINTS)]
                st.dataframe(df)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
        finally:
            os.unlink(tmp_path)
