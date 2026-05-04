import streamlit as st
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Patient Profile", page_icon="🧑")

from app.utils import get_db_connection

ACCENT = "#4b9b79"


def patients_page():
    st.markdown(f'''
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: {ACCENT}; font-weight: 700; margin-bottom: 0.5rem;">Patients</h1>
        <p style="color: #888; font-size: 1.1rem;">Manage patient profiles</p>
    </div>
    ''', unsafe_allow_html=True)

    with st.expander("Add New Patient", expanded=False):
        with st.form("add_patient_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Patient Name", key="new_patient_name")
                dob = st.date_input("Date of Birth", key="new_patient_dob")
                gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="new_patient_gender")
            with col2:
                guardian_name = st.text_input("Guardian Name", key="new_guardian_name")
                guardian_email = st.text_input("Guardian Email", key="new_guardian_email")

            submitted = st.form_submit_button("Add Patient", type="primary")

            if submitted:
                if not name:
                    st.error("Patient name is required")
                else:
                    conn = get_db_connection()
                    conn.execute('''
                        INSERT INTO patients (name, date_of_birth, gender, guardian_name, guardian_email, created_by)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (name, str(dob), gender, guardian_name, guardian_email, st.session_state.get("user_id", 1)))
                    conn.commit()
                    conn.close()
                    st.success(f"Patient '{name}' added successfully!")
                    st.rerun()

    conn = get_db_connection()
    cursor = conn.execute('''
        SELECT p.id, p.name, p.date_of_birth, p.guardian_name, p.guardian_email,
               COUNT(a.id) as assessment_count, MAX(a.created_at) as last_assessment
        FROM patients p
        LEFT JOIN assessments a ON p.id = a.patient_id
        GROUP BY p.id
        ORDER BY p.name
    ''')
    patients = cursor.fetchall()
    conn.close()

    if patients:
        for p in patients:
            patient_id, name, dob, guardian_name, guardian_email, assessment_count, last_date = p

            age = ""
            if dob:
                born = datetime.strptime(str(dob), "%Y-%m-%d")
                today = datetime.now()
                age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))

            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                with col1:
                    st.markdown(f"**{name}**")
                    st.caption(f"Age: {age}" if age else "Age: N/A")
                with col2:
                    st.write(f"**{assessment_count}**")
                    st.caption("Assessments")
                with col3:
                    st.write(f"**{last_date[:10] if last_date else 'N/A'}**")
                    st.caption("Last Visit")
                with col4:
                    st.write(f"**{guardian_name if guardian_name else '-'}**")
                    st.caption("Guardian")
                with col5:
                    if st.button("View Profile", key=f"view_profile_{patient_id}"):
                        st.session_state.current_patient_id = patient_id
                        st.switch_page("pages/patient_profile.py")
                st.divider()
    else:
        st.info("No patients yet. Add your first patient above!")


if __name__ == "__main__":
    patients_page()