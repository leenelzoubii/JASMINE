import streamlit as st
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import get_db_connection

ACCENT = "#4b9b79"


def patient_profile_page():
    patient_id = st.session_state.get("current_patient_id")

    if not patient_id:
        st.error("No patient selected. Please go to Dashboard or Patients and click View.")
        if st.button("Go to Dashboard"):
            st.switch_page("pages/dashboard.py")
        return

    patient_id = int(patient_id)
    conn = get_db_connection()

    cursor = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = cursor.fetchone()

    if not patient:
        st.error("Patient not found")
        conn.close()
        return

    name, dob, gender, guardian_name, guardian_email, created_by, created_at = (
        patient[1], patient[2], patient[3], patient[4], patient[5], patient[6], patient[7]
    )

    age = ""
    if dob:
        born = datetime.strptime(str(dob), "%Y-%m-%d")
        today = datetime.now()
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    st.markdown(f'''
    <div style="padding: 1rem 0;">
        <h1 style="color: {ACCENT}; font-weight: 700;">{name}</h1>
        <p style="color: #666;">DOB: {dob} ({age} years old) | Gender: {gender or 'N/A'}</p>
        <p style="color: #666;">Guardian: {guardian_name or 'N/A'} | {guardian_email or 'N/A'}</p>
    </div>
    ''', unsafe_allow_html=True)

    cursor = conn.execute('''
        SELECT id, inference_score, inference_risk, is_correct, final_diagnosis, status, created_at
        FROM assessments
        WHERE patient_id = ?
        ORDER BY created_at DESC
    ''', (patient_id,))
    assessments = cursor.fetchall()

    if assessments:
        latest = assessments[0]
        assessment_id, score, risk, is_correct, diagnosis, status, created_at = latest

        risk_color = "#d62728" if risk == "High" else ("#ff7f0e" if risk == "Moderate" else "#2ca02c")

        st.markdown("### Inference Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{score:.0%}" if score else "N/A")
        with col2:
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk or 'N/A'}</span>", unsafe_allow_html=True)
        with col3:
            st.write(f"**Status:** {status}")

        with st.expander("Mark Inference as Correct/Incorrect", expanded=False):
            new_is_correct = 1 if is_correct == 1 else 0
            new_is_correct = st.radio(
                "Is the inference result accurate?",
                options=[(1, "Correct"), (0, "Incorrect")],
                format_func=lambda x: x[1],
                index=new_is_correct,
                key="is_correct_radio"
            )

            if st.button("Update Inference Status"):
                conn.execute("UPDATE assessments SET is_correct = ? WHERE id = ?", (new_is_correct[0], assessment_id))
                conn.commit()
                st.success("Updated!")
                st.rerun()

        st.markdown("### Final Diagnosis")
        with st.form("diagnosis_form"):
            new_diagnosis = st.text_area("Professional Diagnosis", value=diagnosis or "", height=100)
            save_diag = st.form_submit_button("Save Diagnosis")

            if save_diag:
                conn.execute(
                    "UPDATE assessments SET final_diagnosis = ?, status = 'reviewed' WHERE id = ?",
                    (new_diagnosis, assessment_id)
                )
                conn.commit()
                st.success("Diagnosis saved!")
                st.rerun()

        if status == "reviewed":
            if st.button("Share with Guardian"):
                conn.execute("UPDATE assessments SET status = 'shared' WHERE id = ?", (assessment_id,))
                conn.commit()
                st.success("Shared with guardian!")
                st.rerun()
    else:
        st.info("No assessments yet. Run inference to create one.")

    st.markdown("---")
    st.markdown("### Notes")

    with st.form(f"add_note_form_{patient_id}"):
        note_text = st.text_area("Add a note", height=80, key=f"new_note_{patient_id}")
        submitted = st.form_submit_button("Save Note")

        if submitted:
            if note_text.strip():
                professional_id = st.session_state.get("user_id", 1)
                conn.execute(
                    "INSERT INTO notes (patient_id, professional_id, content, is_shared) VALUES (?, ?, ?, ?)",
                    (patient_id, professional_id, note_text.strip(), 0)
                )
                conn.commit()
                st.success("Note added!")
                st.rerun()
            else:
                st.error("Note cannot be empty")

    cursor = conn.execute('''
        SELECT id, content, is_shared, created_at
        FROM notes
        WHERE patient_id = ?
        ORDER BY created_at DESC
    ''', (patient_id,))
    notes = cursor.fetchall()

    if notes:
        for note_id, note_content, is_shared, note_created in notes:
            try:
                formatted_date = datetime.strptime(str(note_created), '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y - %I:%M %P')
            except:
                formatted_date = str(note_created)

            with st.container():
                col_left, col_right = st.columns([6, 1])
                with col_left:
                    st.markdown(f"""
                    <div style="background: #f9f9f9; padding: 12px; border-radius: 8px; margin: 8px 0;">
                        <p style="color: #666; font-size: 0.85rem; margin-bottom: 8px;">
                            {formatted_date}
                            {" | Shared with Guardian" if is_shared else " | Private"}
                        </p>
                        <p>{note_content}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_right:
                    with st.popover("Actions"):
                        if st.button("Edit", key=f"edit_note_{note_id}"):
                            st.session_state[f"editing_note_{note_id}"] = True
                            st.rerun()
                        if st.button("Delete", key=f"delete_note_{note_id}"):
                            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
                            conn.commit()
                            st.success("Note deleted!")
                            st.rerun()

            if st.session_state.get(f"editing_note_{note_id}"):
                edit_key = f"edit_text_{note_id}"
                edited_content = st.text_area("Edit note", value=note_content, key=edit_key, height=80)
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Save Changes", key=f"save_edit_{note_id}"):
                        conn.execute("UPDATE notes SET content = ? WHERE id = ?", (edited_content, note_id))
                        conn.commit()
                        st.session_state[f"editing_note_{note_id}"] = False
                        st.success("Note updated!")
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_edit_{note_id}"):
                        st.session_state[f"editing_note_{note_id}"] = False
                        st.rerun()

            st.divider()
    else:
        st.info("No notes yet")

    conn.close()


if __name__ == "__main__":
    patient_profile_page()