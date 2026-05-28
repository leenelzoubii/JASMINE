import os
import sys
import json
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import load_comparison_results

ACCENT = "#4b9b79"


def model_comparison_page():
    st.markdown(f'''
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: {ACCENT}; font-weight: 700; margin-bottom: 0.5rem;">Model Comparison Dashboard</h1>
        <p style="color: #888; font-size: 1.1rem;">Side-by-side comparison of all trained models</p>
    </div>
    ''', unsafe_allow_html=True)

    results_path = PROJECT_ROOT / "models" / "comparison_results.json"
    results = load_comparison_results(str(results_path))

    if results is None:
        st.warning(
            "No comparison results found. Run the training pipeline first:\n"
            "```python\n"
            "python train.py --data_dir /path/to/mmasd\n"
            "```"
        )

        uploaded = st.file_uploader("Or upload comparison_results.json", type=['json'])
        if uploaded:
            results = json.load(uploaded)

    if results:
        view_mode = st.radio(
            "View Mode:",
            ["All Models", "ML Only", "DL Only"],
            horizontal=True,
        )

        st.markdown("### Performance Metrics")
        comparison_data = results.get('comparison', [])

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            if view_mode == "ML Only":
                df = df[df['Model'].isin(['RF', 'SVM'])]
            elif view_mode == "DL Only":
                df = df[df['Model'].isin(['LSTM', 'TRANSFORMER'])]

            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### Accuracy Comparison")
            chart_data = df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].astype(float)
            st.bar_chart(chart_data)

        st.markdown("### Confusion Matrices")
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
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm_array, interpolation='nearest', cmap='Greens')
                ax.set_title(f'{model_type.upper()} Confusion Matrix', fontsize=14, fontweight='bold', color=ACCENT)
                labels = ['TD', 'ASD']
                ax.set_xticks([0, 1])
                ax.set_xticklabels(labels, fontsize=12)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(labels, fontsize=12)

                thresh = cm_array.max() / 2.0
                for ii in range(cm_array.shape[0]):
                    for jj in range(cm_array.shape[1]):
                        value = int(cm_array[ii, jj])
                        ax.text(jj, ii, str(value),
                               ha='center', va='center',
                               color='white' if cm_array[ii, jj] > thresh else ACCENT,
                               fontsize=16, fontweight='bold')

                ax.set_ylabel('True Label', fontsize=12)
                ax.set_xlabel('Predicted Label', fontsize=12)
                plt.tight_layout()

                with cols[i % 2]:
                    st.pyplot(fig)
                    plt.close(fig)

        st.markdown("### Ensemble Results")
        ensemble = results.get('ensemble', {})
        if ensemble:
            e_acc = ensemble.get('accuracy', 0)
            e_f1 = ensemble.get('f1', 0)
            e_auc = ensemble.get('roc_auc', 0)
            e_prec = ensemble.get('precision', 0)
            e_rec = ensemble.get('recall', 0)

            cols = st.columns(5)
            with cols[0]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{e_acc:.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{e_f1:.3f}</div><div class="metric-label">F1 Score</div></div>', unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{e_auc:.3f}</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{e_prec:.3f}</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
            with cols[4]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{e_rec:.3f}</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)

            st.info("**Weighted Ensemble** combines all 4 models using ROC-AUC-based weights, significantly outperforming individual models.")

        st.markdown("### Feature Importance (Random Forest)")
        model_data = models_data.get('rf', {})
        top_features = model_data.get('top_features', {})

        if top_features:
            from src.visualization.plots import plot_feature_importance
            feat_names = list(top_features.keys())
            feat_values = np.array(list(top_features.values()))
            fig = plot_feature_importance(feat_names, feat_values, top_n=15)
            st.pyplot(fig)
            plt.close(fig)


if __name__ == "__main__":
    model_comparison_page()