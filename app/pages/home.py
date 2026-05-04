import streamlit as st

ACCENT = "#4b9b79"


def home_page():
    st.markdown(f'''
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: {ACCENT}; font-weight: 700; margin-bottom: 0.5rem;">Autism Spectrum Disorder Screening</h1>
        <p style="color: #888; font-size: 1.1rem;">Privacy-Preserving Analysis via 2D Pose Estimation</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("### How It Works")
    flow_col = st.container()
    with flow_col:
        st.markdown(
            f'''
            <div style="text-align:center; padding: 20px;">
                <span class="flow-step">Video Input</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">OpenPose 2D</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">Keypoints (25 joints)</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">Feature Extraction</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">ML/DL Models</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">Prediction</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.markdown("### Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'''
            <div class="metric-card">
                <div class="metric-value">25</div>
                <div class="metric-label">BODY_25 Keypoints</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '''
            <div class="metric-card">
                <div class="metric-value">4</div>
                <div class="metric-label">Models Compared</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '''
            <div class="metric-card">
                <div class="metric-value">200+</div>
                <div class="metric-label">Extracted Features</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '''
            <div class="metric-card">
                <div class="metric-value">100%</div>
                <div class="metric-label">Privacy Preserving</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Feature Types")
        st.markdown(
            f'''
            <div class="info-box">
                <b>Kinematic Features</b><br>
                Joint angles, velocities, inter-joint distances, 
                and body symmetry metrics extracted from pose sequences.
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
            <div class="info-box">
                <b>Statistical Features</b><br>
                Per-joint coordinate statistics (mean, std, min, max, median, range) 
                across all frames.
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
            <div class="info-box">
                <b>Temporal & Frequency Features</b><br>
                Frame-to-frame differences, autocorrelation, and FFT power spectrum 
                analysis for movement pattern detection.
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### Models")
        st.markdown(
            f'''
            <div class="info-box">
                <b>Random Forest</b><br>
                Ensemble of decision trees with feature importance. 
                Robust to overfitting, provides interpretable results.
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
            <div class="info-box">
                <b>SVM (Support Vector Machine)</b><br>
                Kernel-based classifier effective for high-dimensional feature spaces. 
                With RBF and linear kernels.
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
            <div class="info-box">
                <b>LSTM (Long Short-Term Memory)</b><br>
                Recurrent neural network that captures temporal dependencies 
                in pose sequences. Bidirectional with 2 layers.
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
            <div class="info-box">
                <b>Transformer</b><br>
                Self-attention based model for sequence classification. 
                Captures long-range temporal patterns with positional encoding.
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.warning(
        "**Privacy Note:** This system processes only 2D skeletal keypoints "
        "(x, y coordinates). No raw video frames, images, or personally identifiable "
        "visual data are stored or transmitted."
    )