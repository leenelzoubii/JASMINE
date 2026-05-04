import streamlit as st
from datetime import datetime
from pathlib import Path
import sys

st.set_page_config(page_title="Patient Profile", page_icon="🧑")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import get_db_connection

ACCENT = "#4b9b79"


def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


def get_formatted_datetime():
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y - %I:%M %p")


def dashboard_page():
    professional_name = st.session_state.professional_name or st.session_state.username

    st.markdown(f'''
    <div style="padding: 1rem 0;">
        <h1 style="color: {ACCENT}; font-weight: 700; margin-bottom: 0.5rem;">{get_greeting()}, {professional_name}!</h1>
        <p style="color: #666; font-size: 1.1rem;">{get_formatted_datetime()}</p>
    </div>
    ''', unsafe_allow_html=True)

    conn = get_db_connection()

    cursor = conn.execute("SELECT COUNT(*) FROM assessments WHERE status = 'pending'")
    pending_count = cursor.fetchone()[0]

    if pending_count > 0:
        st.markdown(f'''
        <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; border-radius: 4px; margin-bottom: 1.5rem;">
            <b>🔔 {pending_count} assessment{"s" if pending_count > 1 else ""} awaiting review</b>
        </div>
        ''', unsafe_allow_html=True)

    cursor = conn.execute("SELECT COUNT(*) FROM assessments WHERE created_at >= date('now', '-7 days')")
    week_count = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM assessments")
    total_count = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM notes WHERE is_shared = 0")
    pending_notes = cursor.fetchone()[0]

    conn.close()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{week_count}</div>
            <div class="metric-label">This Week</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{total_count}</div>
            <div class="metric-label">Total Assessments</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: #ff7f0e;">{pending_count}</div>
            <div class="metric-label">Pending Review</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: #d62728;">{pending_notes}</div>
            <div class="metric-label">Pending Notes</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2 = st.tabs(["Recent Patients", "Pending Review"])

    with tab1:
        conn = get_db_connection()
        cursor = conn.execute('''
            SELECT p.id, p.name, p.date_of_birth, p.gender,
                   a.inference_score, a.inference_risk, a.final_diagnosis, a.status, a.created_at
            FROM patients p
            LEFT JOIN (
                SELECT patient_id, inference_score, inference_risk, final_diagnosis, status, created_at
                FROM assessments
                WHERE id IN (SELECT MAX(id) FROM assessments GROUP BY patient_id)
            ) a ON p.id = a.patient_id
            ORDER BY a.created_at DESC
        ''')
        patients = cursor.fetchall()
        conn.close()

        if patients:
            for p in patients:
                patient_id, name, dob, gender, score, risk, diagnosis, status, created_at = p
                risk_color = "#d62728" if risk == "High" else ("#ff7f0e" if risk == "Moderate" else "#2ca02c")
                risk_display = risk or "N/A"
                score_display = f"{score:.0%}" if score else "N/A"
                diagnosis_display = diagnosis if diagnosis else "Pending"

                with st.container():
                    col_left, col_right = st.columns([6, 1])
                    with col_left:
                        st.markdown(f"**{name}**")
                        st.caption(f"Risk: {risk_display} | Score: {score_display} | Diagnosis: {diagnosis_display}")
                    with col_right:
                        if st.button("View", key=f"view_patient_{patient_id}"):
                            st.session_state.current_patient_id = patient_id
                            st.switch_page("pages/patient_profile.py")
                    st.divider()
        else:
            st.info("No patients yet. Go to Patients to add one.")

    with tab2:
        conn = get_db_connection()
        cursor = conn.execute('''
            SELECT a.id, a.inference_score, a.inference_risk, a.created_at, p.name, p.id
            FROM assessments a
            JOIN patients p ON a.patient_id = p.id
            WHERE a.status = 'pending'
            ORDER BY a.created_at DESC
        ''')
        pending_assessments = cursor.fetchall()
        conn.close()

        if pending_assessments:
            for a in pending_assessments:
                assessment_id, score, risk, created_at, patient_name, patient_id = a
                risk_color = "#ff7f0e" if risk == "Moderate" else "#d62728"
                with st.container():
                    col_left, col_right = st.columns([6, 1])
                    with col_left:
                        st.markdown(f"**{patient_name}**")
                        st.caption(f"Score: {score:.0%} | Risk: {risk} | {created_at}")
                    with col_right:
                        if st.button("Review", key=f"review_patient_{patient_id}"):
                            st.session_state.current_patient_id = patient_id
                            st.switch_page("pages/patient_profile.py")
                    st.divider()
        else:
            st.success("No pending assessments to review!")


if __name__ == "__main__":
    dashboard_page()