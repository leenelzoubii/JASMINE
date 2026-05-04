import os
import sys
import json
import email.utils
from pathlib import Path

import streamlit as st
import bcrypt
import sqlite3

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="JASMINE - Autism Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ SESSION STATE ============
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "role" not in st.session_state:
    st.session_state.role = ""

if "professional_name" not in st.session_state:
    st.session_state.professional_name = ""

if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

# ============ WELCOME PAGE PLACEHOLDERS ============
# Edit these to customize the welcome page

WELCOME_TITLE = "JASMINE"
WELCOME_TITLE_COLOR = "#4b9b79"  # Change this - main title color

WELCOME_SUBTITLE = "Joint Analysis and Screening for Motor Imbalances"
WELCOME_SUBTITLE_COLOR = "#888888"  # Change this - subtitle color

WELCOME_DESCRIPTION = """JASMINE is a privacy-preserving autism spectrum disorder 
screening system that uses 2D pose estimation to analyze movement patterns."""

# Feature cards - edit title, description, and icon for each
FEATURES = [
    {
        "title": "Pose Estimation",
        "description": "Advanced 2D pose detection using BODY-25 keypoints",
        "icon": "🧠",
    },
    {
        "title": "Multi-Model Analysis",
        "description": "Compare predictions from Random Forest, SVM, LSTM, and Transformer",
        "icon": "📊",
    },
    {
        "title": "Privacy First",
        "description": "Only skeletal keypoints processed - no raw videos or images stored",
        "icon": "🔒",
    },
    {
        "title": "Interactive Visualization",
        "description": "Visualize skeletons with bounding boxes and heatmaps",
        "icon": "📈",
    },
]

CTA_MESSAGE = "Ready to begin screening?"
CTA_BUTTON_TEXT = "Get Started"

# ============ COLORS ============
ACCENT = "#4b9b79"
WHITE = "#ffffff"
BLACK = "#000000"
GRAY_LIGHT = "#f5f5f5"
GRAY_MEDIUM = "#888888"

# ============ CSS STYLES ============
st.markdown(f"""
<style>
    .stApp {{ background: {WHITE}; }}
    h1, h2, h3 {{ color: {BLACK}; font-weight: 600; }}
    h1 {{ font-size: 2.2rem; text-align: center; padding: 1rem 0; }}
    h2 {{ font-size: 1.5rem; margin-bottom: 1rem; }}
    h3 {{ font-size: 1.2rem; }}

    .main-header {{ font-size: 2.5rem; font-weight: 700; color: {ACCENT}; text-align: center; margin-bottom: 1rem; }}
    .sub-header {{ font-size: 1.2rem; color: {GRAY_MEDIUM}; text-align: center; margin-bottom: 2rem; }}

    .metric-card {{ background: {WHITE}; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(75, 155, 121, 0.15); text-align: center; border: 1px solid #e0e0e0; transition: all 0.2s ease; }}
    .metric-card:hover {{ box-shadow: 0 4px 20px rgba(75, 155, 121, 0.25); transform: translateY(-2px); }}
    .metric-value {{ font-size: 2rem; font-weight: 700; color: {ACCENT}; }}
    .metric-label {{ font-size: 0.9rem; color: {GRAY_MEDIUM}; margin-top: 4px; }}

    .info-box {{ background: {GRAY_LIGHT}; border-radius: 10px; padding: 18px; border-left: 4px solid {ACCENT}; margin: 12px 0; transition: all 0.2s ease; }}
    .info-box:hover {{ border-left-width: 6px; box-shadow: 0 2px 12px rgba(75, 155, 121, 0.15); }}

    .flow-step {{ display: inline-block; background: {ACCENT}; color: {WHITE}; padding: 10px 18px; border-radius: 22px; margin: 4px; font-size: 0.85rem; font-weight: 500; transition: all 0.2s ease; }}
    .flow-step:hover {{ transform: scale(1.05); box-shadow: 0 4px 12px rgba(75, 155, 121, 0.3); }}
    .flow-arrow {{ display: inline-block; color: {GRAY_MEDIUM}; font-size: 1.2rem; margin: 0 4px; }}

    .model-card {{ background: {WHITE}; border-radius: 10px; padding: 16px; box-shadow: 0 1px 8px rgba(0,0,0,0.08); margin-bottom: 12px; border: 1px solid #e0e0e0; }}
    .risk-low {{ color: #2ca02c; font-weight: 700; }}
    .risk-moderate {{ color: #ff7f0e; font-weight: 700; }}
    .risk-high {{ color: #d62728; font-weight: 700; }}

    .login-container {{ max-width: 420px; margin: 0 auto; padding: 0; background: transparent; }}

    /* Login page dramatic styling */
    .login-page-bg {{
        background: linear-gradient(135deg, #4b9b79 0%, #2d6b52 50%, #1a4a38 100%);
        min-height: 100vh;
        padding: 2rem;
    }}

    .login-card {{
        background: white;
        border-radius: 20px;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.3);
        border: 2px solid white;
        overflow: hidden;
    }}

    .login-card-header {{
        background: linear-gradient(135deg, #4b9b79 0%, #5ba88a 100%);
        padding: 2rem;
        text-align: center;
    }}

    .login-card-title {{
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }}

    .login-card-subtitle {{
        color: rgba(255, 255, 255, 0.85);
        font-size: 1rem;
    }}

    .login-card-body {{
        padding: 2rem;
    }}

    .login-divider {{
        height: 3px;
        background: linear-gradient(to right, #4b9b79, #8fd4b3, #4b9b79);
        margin: 0;
    }}

    /* Login/Register radio toggle */
    .login-toggle {{
        display: flex;
        background: #f0f0f0;
        border-radius: 10px;
        padding: 4px;
        margin-bottom: 1.5rem;
    }}

    .login-toggle label {{
        flex: 1;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 500;
    }}

    .login-toggle input:checked + label {{
        background: #4b9b79;
        color: white;
    }}

    .login-toggle input {{
        display: none;
    }}

    /* Form inputs */
    .login-input label {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 6px;
        display: block;
}}

    .login-input input {{
        width: 100%;
        padding: 12px 16px;
        border: 2px solid #ddd;
        border-radius: 10px;
        font-size: 1rem;
        transition: all 0.2s ease;
    }}

    .login-input input:focus {{
        border-color: #4b9b79;
        box-shadow: 0 0 0 3px rgba(75, 155, 121, 0.15);
        outline: none;
    }}

    .login-btn {{
        width: 100%;
        padding: 14px;
        background: #4b9b79;
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 1rem;
    }}

    .login-btn:hover {{
        background: #3d8a6a;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(75, 155, 121, 0.35);
    }}

    /* Expander */
    .login-bypass {{
        margin-top: 1.5rem;
    }}

    /* Buttons */
    .stButton > button {{ background: {ACCENT}; color: {WHITE}; border: none; border-radius: 8px; padding: 0.6rem 1.5rem; font-weight: 600; transition: all 0.2s ease; }}
    .stButton > button:hover {{ background: #3d8a6a; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(75, 155, 121, 0.3); }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background: {GRAY_LIGHT}; border-right: 1px solid #e0e0e0; }}
    .stRadio > div {{ gap: 0.5rem; }}

    /* Inputs */
    .stTextInput > div > div > input {{ border-radius: 8px; border: 1px solid #ddd; padding: 0.5rem 1rem; transition: all 0.2s ease; }}
    .stTextInput > div > div > input:focus {{ border-color: {ACCENT}; box-shadow: 0 0 0 2px rgba(75, 155, 121, 0.2); }}

    /* Messages */
    .stSuccess {{ background: #d4edda; color: #155724; border-radius: 8px; border-left: 4px solid {ACCENT}; }}
    .stError {{ background: #f8d7da; color: #721c24; border-radius: 8px; border-left: 4px solid #dc3545; }}
    .stWarning {{ background: #fff3cd; color: #856404; border-radius: 8px; border-left: 4px solid #ffc107; }}
    .stInfo {{ background: #d1ecf1; color: #0c5460; border-radius: 8px; border-left: 4px solid {ACCENT}; }}

    /* Progress */
    .stProgress > div > div > div {{ background: {ACCENT}; }}

    /* Divider */
    hr {{ border: none; height: 2px; background: linear-gradient(to right, {ACCENT}, transparent); margin: 1.5rem 0; }}

    /* Welcome page styles */
    .welcome-container {{ max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }}
    .welcome-title {{ font-size: 4rem; font-weight: 800; text-align: center; margin-bottom: 0.5rem; letter-spacing: 4px; }}
    .welcome-subtitle {{ font-size: 1.5rem; text-align: center; margin-bottom: 2rem; }}
    .welcome-description {{ font-size: 1.1rem; text-align: center; color: {GRAY_MEDIUM}; max-width: 600px; margin: 0 auto 3rem; line-height: 1.6; }}
    .feature-card {{ background: {WHITE}; border-radius: 16px; padding: 24px; box-shadow: 0 2px 16px rgba(0,0,0,0.08); border: 1px solid #e0e0e0; text-align: center; transition: all 0.2s ease; height: 100%; }}
    .feature-card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 24px rgba(75, 155, 121, 0.2); border-color: {ACCENT}; }}
    .feature-icon {{ font-size: 2.5rem; margin-bottom: 1rem; }}
    .feature-title {{ font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; }}
    .feature-description {{ font-size: 0.95rem; color: {GRAY_MEDIUM}; }}
    .cta-container {{ text-align: center; margin-top: 3rem; padding: 2rem; }}
    .cta-message {{ font-size: 1.3rem; margin-bottom: 1rem; }}
    .get-started-btn {{ background: {ACCENT}; color: {WHITE}; border: none; border-radius: 8px; padding: 0.8rem 2rem; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: all 0.2s ease; }}
    .get-started-btn:hover {{ background: #3d8a6a; transform: scale(1.02); box-shadow: 0 4px 16px rgba(75, 155, 121, 0.3); }}
    .top-left-btn {{ position: fixed; top: 20px; left: 20px; z-index: 1000; }}
    .get-started-btn-small {{ background: {ACCENT}; color: {WHITE}; border: none; border-radius: 6px; padding: 0.5rem 1.2rem; font-size: 0.95rem; font-weight: 600; cursor: pointer; }}
    .get-started-btn-small:hover {{ background: #3d8a6a; }}

    /* Fixed top navigation */
    .fixed-nav {{ position: fixed; top: 0; left: 0; right: 0; height: 60px; background: {ACCENT}; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; z-index: 9999; color: white; box-shadow: 0 2px 12px rgba(0,0,0,0.15); }}
    .nav-brand {{ font-size: 1.4rem; font-weight: 700; letter-spacing: 1px; }}
    .nav-links {{ display: flex; gap: 8px; }}
    .nav-links a {{ color: white; text-decoration: none; padding: 8px 14px; border-radius: 6px; transition: all 0.2s; font-weight: 500; }}
    .nav-links a:hover, .nav-links a.active {{ background: rgba(255,255,255,0.25); }}
    .nav-user {{ display: flex; align-items: center; gap: 16px; }}
    .nav-logout-btn {{ background: rgba(255,255,255,0.2); color: white; border: none; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-weight: 500; transition: all 0.2s; }}
    .nav-logout-btn:hover {{ background: rgba(255,255,255,0.35); }}
    .main-content {{ margin-top: 76px; }}
    .no-nav {{ margin-top: 0; }}
</style>
""", unsafe_allow_html=True)


# ============ FUNCTIONS ============
def get_db_connection():
    DB_PATH = PROJECT_ROOT / "users.db"
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH = PROJECT_ROOT / "users.db"

    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('professional', 'guardian')),
            professional_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date_of_birth TEXT,
            gender TEXT,
            guardian_name TEXT,
            guardian_email TEXT,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            professional_id INTEGER NOT NULL,
            inference_score REAL,
            inference_risk TEXT,
            is_correct INTEGER,
            final_diagnosis TEXT,
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'reviewed', 'shared')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (professional_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            professional_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            is_shared INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (professional_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

    seed_demo_data()


def load_users():
    init_db()
    conn = get_db_connection()
    cursor = conn.execute("SELECT id, username, email, password_hash, role, professional_name FROM users")
    users = {row["username"]: {"id": row["id"], "email": row["email"], "password": row["password_hash"], "role": row["role"], "professional_name": row["professional_name"]} for row in cursor}
    conn.close()
    return users


def find_user_by_identifier(identifier):
    init_db()
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT id, username, email, password_hash, role, professional_name FROM users WHERE username = ? OR email = ?",
        (identifier, identifier)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row["id"], "username": row["username"], "email": row["email"], "password": row["password_hash"], "role": row["role"], "professional_name": row["professional_name"]}
    return None


def get_user_by_id(user_id):
    init_db()
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT id, username, email, role, professional_name FROM users WHERE id = ?",
        (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row["id"], "username": row["username"], "email": row["email"], "role": row["role"], "professional_name": row["professional_name"]}
    return None


def save_user(username, email, password, role, professional_name=None):
    if not username or not email or not password or not role:
        return False
    conn = get_db_connection()
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash, role, professional_name) VALUES (?, ?, ?, ?, ?)",
            (username, email, password_hash, role, professional_name)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def verify_password(password, stored_hash):
    return bcrypt.checkpw(password.encode(), stored_hash.encode())


def seed_demo_data():
    conn = get_db_connection()

    cursor = conn.execute("SELECT COUNT(*) as count FROM users WHERE username = 'drjasmine'")
    if cursor.fetchone()[0] == 0:
        password_hash = bcrypt.hashpw("demo123".encode(), bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT INTO users (username, email, password_hash, role, professional_name) VALUES (?, ?, ?, ?, ?)",
            ("drjasmine", "dr.jasmine@jasmine.com", password_hash, "professional", "Dr. Jasmine")
        )

        cursor = conn.execute("SELECT id FROM users WHERE username = 'drjasmine'")
        professional_id = cursor.fetchone()[0]

        patients_data = [
            ("Emma Thompson", "2018-03-15", "Female", "John Thompson", "john.thompson@email.com"),
            ("Liam Johnson", "2019-07-22", "Male", "Sarah Johnson", "sarah.j@email.com"),
            ("Sophie Williams", "2017-11-08", "Female", "Mike Williams", "mike.w@email.com"),
        ]

        for name, dob, gender, guardian_name, guardian_email in patients_data:
            conn.execute(
                "INSERT INTO patients (name, date_of_birth, gender, guardian_name, guardian_email, created_by) VALUES (?, ?, ?, ?, ?, ?)",
                (name, dob, gender, guardian_name, guardian_email, professional_id)
            )

        cursor = conn.execute("SELECT id FROM patients ORDER BY id")
        patient_ids = [row[0] for row in cursor.fetchall()]

        assessments_data = [
            (patient_ids[0], professional_id, 0.82, "High", 1, "ASD - Level 2 Support Required", "shared"),
            (patient_ids[1], professional_id, 0.58, "Moderate", 0, None, "pending"),
            (patient_ids[2], professional_id, 0.23, "Low", 1, "No ASD indicators", "reviewed"),
        ]

        for patient_id, prof_id, score, risk, is_correct, diagnosis, status in assessments_data:
            conn.execute(
                "INSERT INTO assessments (patient_id, professional_id, inference_score, inference_risk, is_correct, final_diagnosis, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (patient_id, prof_id, score, risk, is_correct, diagnosis, status)
            )

        notes_data = [
            (patient_ids[0], professional_id, "Initial assessment completed. Patient showed limited eye contact and repetitive movements.", 0),
            (patient_ids[0], professional_id, "Follow-up session: Improvement observed in social engagement.", 1),
            (patient_ids[1], professional_id, "Assessment pending. Need more time to observe behavior patterns.", 0),
            (patient_ids[2], professional_id, "Low risk indicators confirmed. Recommend annual screening.", 1),
        ]

        for patient_id, prof_id, content, is_shared in notes_data:
            conn.execute(
                "INSERT INTO notes (patient_id, professional_id, content, is_shared) VALUES (?, ?, ?, ?)",
                (patient_id, prof_id, content, is_shared)
            )

        conn.commit()

    conn.close()


def is_valid_email(email_address):
    import email.utils as email_utils
    parsed = email_utils.parseaddr(email_address)
    return bool(parsed[1] and "@" in parsed[1])


def show_welcome_page():
    # Reset background to white for welcome page
    st.markdown("<style>.stApp { background: white !important; }</style>", unsafe_allow_html=True)

    # Get Started button in top-right - use columns to push to right
    col_space, col_btn = st.columns([9, 1])
    with col_btn:
        if st.button("Get Started", key="welcome_get_started"):
            st.session_state.welcome_shown = True
            st.rerun()

    st.markdown(f'''
    <div class="welcome-container">
        <div class="welcome-title" style="color: {WELCOME_TITLE_COLOR};">{WELCOME_TITLE}</div>
        <div class="welcome-subtitle" style="color: {WELCOME_SUBTITLE_COLOR};">{WELCOME_SUBTITLE}</div>
        <div class="welcome-description">{WELCOME_DESCRIPTION}</div>
    </div>
    ''', unsafe_allow_html=True)

    # Feature cards
    cols = st.columns(len(FEATURES))
    for i, feature in enumerate(FEATURES):
        with cols[i]:
            st.markdown(f'''
            <div class="feature-card">
                <div class="feature-icon">{feature["icon"]}</div>
                <div class="feature-title">{feature["title"]}</div>
                <div class="feature-description">{feature["description"]}</div>
            </div>
            ''', unsafe_allow_html=True)

    # CTA
    st.markdown(f'''
    <div class="cta-container">
        <div class="cta-message">{CTA_MESSAGE}</div>
    </div>
    ''', unsafe_allow_html=True)

    if st.button(CTA_BUTTON_TEXT, key="get_started"):
        st.session_state.welcome_shown = True
        st.rerun()


def show_login_page():
    pass

    # Centered white card
    col_left, col_center, col_right = st.columns([1, 1.8, 1])

    with col_center:
        # White card with everything inside
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; 
                    box-shadow: 0 8px 32px rgba(0,0,0,0.25); margin-top: 2rem; margin-bottom: 2rem;">
            <h2 style="color: #4b9b79; margin: 0 0 0.5rem; text-align: center; font-weight: 700;">Welcome to JASMINE</h2>
            <p style="color: #888; margin: 0 0 1.5rem; text-align: center;">Sign in to continue</p>
        </div>
        """, unsafe_allow_html=True)

        # Login/Register toggle - check pending state before rendering
        if "switch_to_register" not in st.session_state:
            st.session_state.switch_to_register = False

        if "login_menu" not in st.session_state:
            st.session_state.login_menu = "Login"

        # Check if we need to switch to register tab (before radio renders)
        if st.session_state.switch_to_register:
            st.session_state.login_menu = "Register"
            st.session_state.switch_to_register = False

        menu = st.radio("", ["Login", "Register"], horizontal=True, key="login_menu", label_visibility="hidden")
        users = load_users()

        if menu == "Register":
            st.markdown("<p style='font-weight: 600; margin-bottom: 0.5rem;'>Create Account</p>", unsafe_allow_html=True)
            new_username = st.text_input("Username", key="register_username")
            new_email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")

            st.markdown("<p style='font-weight: 600; margin-bottom: 0.5rem;'>Account Type</p>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                professional_selected = st.button(
                    "🏥 Professional",
                    key="btn_professional",
                    use_container_width=True,
                    type="primary" if st.session_state.get("selected_role") == "professional" else "secondary"
                )
            with col2:
                guardian_selected = st.button(
                    "👨‍👩‍👧 Guardian",
                    key="btn_guardian",
                    use_container_width=True,
                    type="primary" if st.session_state.get("selected_role") == "guardian" else "secondary"
                )

            if professional_selected:
                st.session_state.selected_role = "professional"
            elif guardian_selected:
                st.session_state.selected_role = "guardian"

            selected_role = st.session_state.get("selected_role", "")

            if st.button("Create Account", key="register_btn", use_container_width=True):
                if not new_username or not new_email or not new_password:
                    st.error("All fields are required")
                elif not selected_role:
                    st.error("Please select an account type")
                elif not is_valid_email(new_email):
                    st.error("Please enter a valid email address")
                elif new_username in users:
                    st.error("Username already exists")
                else:
                    prof_name = new_username if selected_role == "professional" else None
                    if save_user(new_username, new_email, new_password, selected_role, prof_name):
                        user = find_user_by_identifier(new_username)
                        st.session_state.logged_in = True
                        st.session_state.username = user["username"]
                        st.session_state.user_id = user["id"]
                        st.session_state.role = user["role"]
                        st.session_state.professional_name = user["professional_name"] or ""
                        st.session_state.welcome_shown = True
                        st.session_state.selected_role = ""
                        st.success("Account created!")
                        st.rerun()
                    else:
                        st.error("Email or username already exists")

        elif menu == "Login":
            st.markdown("<p style='font-weight: 600; margin-bottom: 0.5rem;'>Sign In</p>", unsafe_allow_html=True)
            identifier = st.text_input("Username or Email", key="login_identifier")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Sign In", key="login_btn", use_container_width=True):
                if not identifier or not password:
                    st.error("All fields are required")
                else:
                    user = find_user_by_identifier(identifier)
                    if not user:
                        st.session_state.switch_to_register = True
                        st.error(f"No account found with '{identifier}'. Please create one.")
                        st.rerun()
                    elif verify_password(password, user["password"]):
                        st.session_state.logged_in = True
                        st.session_state.username = user["username"]
                        st.session_state.user_id = user["id"]
                        st.session_state.role = user["role"]
                        st.session_state.professional_name = user["professional_name"] or ""
                        st.session_state.welcome_shown = True
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid password")

        with st.expander("Developer Bypass"):
            st.warning("For testing only")
            if st.button("Skip Login (Dev)", key="dev_bypass"):
                init_db()
                conn = get_db_connection()
                cursor = conn.execute("SELECT id FROM users WHERE username = 'drjasmine'")
                row = cursor.fetchone()
                conn.close()
                user_id = row[0] if row else 1

                st.session_state.logged_in = True
                st.session_state.username = "drjasmine"
                st.session_state.user_id = user_id
                st.session_state.role = "professional"
                st.session_state.professional_name = "Dr. Jasmine"
                st.session_state.welcome_shown = True
                st.success("Dev bypass - logged in!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ============ MAIN FLOW ============
# Check if user is logged in
if not st.session_state.logged_in:
    # Not logged in - check if welcome should be shown
    if not st.session_state.welcome_shown:
        # Show welcome page
        show_welcome_page()
    else:
        # Show login page
        show_login_page()
else:
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")

    if st.session_state.role == "professional":
        professional_name = st.session_state.professional_name or st.session_state.username
        st.sidebar.success(f"Welcome, {professional_name}")
        pages = ["Dashboard", "Patients", "Run Inference", "Pose Viewer", "Model Comparison"]
    else:
        role_display = "Guardian"
        st.sidebar.success(f"Welcome, {st.session_state.username} ({role_display})")
        pages = ["Home", "Run Inference", "Pose Viewer", "Model Comparison"]

    page = st.sidebar.radio("Go to:", pages)

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_id = None
        st.session_state.role = ""
        st.session_state.professional_name = ""
        st.session_state.welcome_shown = False
        st.session_state.current_patient_id = None
        st.rerun()

    if page == "Dashboard":
        from app.pages import dashboard
        dashboard.dashboard_page()
    elif page == "Patients":
        from app.pages import patients
        patients.patients_page()
    elif page == "Home":
        from app.pages import home
        home.home_page()
    elif page == "Model Comparison":
        from app.pages import model_comparison
        model_comparison.model_comparison_page()
    elif page == "Run Inference":
        from app.pages import inference
        inference.inference_page()
    elif page == "Pose Viewer":
        from app.pages import pose_viewer
        pose_viewer.pose_viewer_page()