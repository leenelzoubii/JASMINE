import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import load_all_models, get_ensemble_prediction, format_prediction_result, generate_report, get_db_connection, load_ensemble_weights
from src.data.loader import load_openpose_json, load_csv_sequence
from src.features.kinematic import extract_kinematic_features
from src.features.statistical import extract_all_features

ACCENT = "#4b9b79"


def inference_page():
    st.markdown(f'''
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: {ACCENT}; font-weight: 700; margin-bottom: 0.5rem;">Run Inference</h1>
        <p style="color: #888; font-size: 1.1rem;">Upload pose data and get predictions from all models</p>
    </div>
    ''', unsafe_allow_html=True)

    if st.session_state.role != "professional":
        st.warning("This feature is only available for professionals.")
        return

    conn = get_db_connection()
    cursor = conn.execute("SELECT id, name FROM patients ORDER BY name")
    patients = cursor.fetchall()
    conn.close()

    patient_options = ["Select a patient..."] + [f"{p[1]} (ID: {p[0]})" for p in patients]
    selected_patient = st.selectbox("Select Patient", patient_options, key="inference_patient_select")

    if selected_patient == "Select a patient...":
        st.error("Please select a patient before running inference")
        st.stop()

    patient_id = int(selected_patient.split("(ID: ")[1].rstrip(")"))

    models_dir = PROJECT_ROOT / "models"
    models = load_all_models(str(models_dir))

    if not models:
        st.warning(
            "No trained models found. Train models first or upload them.\n"
            "Expected files: `rf_model.pkl`, `svm_model.pkl`, `lstm_model.pth`, `transformer_model.pth`"
        )

    uploaded_file = st.file_uploader(
        "Upload OpenPose JSON or MMASD CSV file",
        type=['json', 'csv'],
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            if uploaded_file.name.endswith('.json'):
                kp = load_openpose_json(tmp_path)
                keypoints_sequence = kp[np.newaxis, ...]
                st.info(f"Loaded OpenPose JSON: {kp.shape[0]} joints detected")

            elif uploaded_file.name.endswith('.csv'):
                kp, action_label, asd_label = load_csv_sequence(tmp_path)
                keypoints_sequence = kp
                st.info(
                    f"Loaded MMASD CSV: {kp.shape[0]} frames, {kp.shape[1]} joints. "
                    f"{'ASD Label: ' + str(asd_label) if asd_label is not None else ''}"
                )

            coords_2d = keypoints_sequence[:, :, :2]
            kinematic_feats, kinematic_names = extract_kinematic_features(coords_2d)
            stat_feats, stat_names = extract_all_features(coords_2d)

            all_features = np.concatenate([kinematic_feats, stat_feats])
            all_names = kinematic_names + stat_names

            dl_sequence = keypoints_sequence.reshape(
                keypoints_sequence.shape[0], -1
            )

            if models:
                predictions = get_ensemble_prediction(models, all_features, dl_sequence)
                weights_path = PROJECT_ROOT / "models" / "comparison_results.json"
                weights = load_ensemble_weights(str(weights_path))
                formatted = format_prediction_result(predictions, weights)

                st.markdown("---")
                st.markdown("### Prediction Results")

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

                st.markdown("### Individual Model Predictions")
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

                inference_score = formatted['ensemble_probability']
                risk = formatted['risk_level'].replace(" Risk", "")

                conn = get_db_connection()
                professional_id = st.session_state.user_id or 1
                conn.execute('''
                    INSERT INTO assessments (patient_id, professional_id, inference_score, inference_risk, is_correct, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (patient_id, professional_id, inference_score, risk, 1, 'pending'))
                conn.commit()
                conn.close()

                st.success(f"Assessment saved for patient! View it in their profile.")

                report = generate_report(uploaded_file.name, formatted)
                report += f"\nEnsemble Weights: { {k: f'{v:.1%}' for k, v in weights.items()} }\n"
                st.markdown("### Prediction Report")
                st.text_area("Report", report, height=300)

                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"prediction_report_{uploaded_file.name}.txt",
                    mime="text/plain",
                )
            else:
                st.info("Models not loaded. Showing feature extraction only.")

                st.markdown("### Extracted Features")
                st.write(f"Total features: {len(all_features)}")
                st.write("Top 20 features by magnitude:")
                top_indices = np.argsort(np.abs(all_features))[::-1][:20]
                for idx in top_indices:
                    st.write(f"  {all_names[idx]}: {all_features[idx]:.4f}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    inference_page()