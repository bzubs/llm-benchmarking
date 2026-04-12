import streamlit as st
import requests
import base64
import os
from config_loader import load_config

st.set_page_config(
    page_title="Neysa Benchmarking Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)

config = load_config()
router_port = config.router_port
router_url = f"http://localhost:{router_port}"


def register_user(username, password):
    try:
        payload = {
            "username": username,
            "password": password,
        }

        res = requests.post(f"{router_url}/register", json=payload)

        if res.status_code == 200:
            return True, res.json().get("message", "Registered successfully")

        return False, res.json().get("detail", "Registration failed")

    except Exception as e:
        return False, str(e)


def login_user(username, password):
    try:
        payload = {"username": username, "password": password}

        res = requests.post(f"{router_url}/login", json=payload)

        if res.status_code == 200:
            data = res.json()
            st.session_state.token = data["access_token"]
            return True, data.get("message", "Login successful")

        return False, res.json().get("detail", "Login failed")

    except Exception as e:
        return False, str(e)


@st.cache_data
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def is_valid_neysa_username(username: str) -> bool:
    username = username.strip().lower()
    return (
        username.endswith("@neysa.ai")
        and "@" in username
        and len(username) > len("@neysa.ai")
    )


def password_strength(password: str) -> str:
    if len(password) >= 10:
        return "Strong"
    if len(password) >= 8:
        return "Good"
    if len(password) >= 6:
        return "Weak"
    return "Very weak"


def show_auth_screen():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: #0a0a0f;
        }

        header {
            visibility: hidden !important;
            height: 0 !important;
        }

        #MainMenu {
            visibility: hidden !important;
        }

        footer {
            visibility: hidden !important;
        }

        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }

        [data-testid="stMainBlockContainer"] {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding-top: 0 !important;
        }

        div[data-testid="stTextInput"] input {
            background-color: #111827 !important;
            color: #f9fafb !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            padding: 10px 12px !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color: #6b7280 !important;
        }

        div[data-testid="stButton"] button {
            border-radius: 10px !important;
            font-weight: 500 !important;
        }

        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #7c6df2 !important;
            height: 44px !important;
        }

        .auth-title {
            font-size: 20px;
            letter-spacing: 5px;
            color: #9ca3af;
            margin-top: 10px;
            text-align: center;
            font-weight: 600;
        }

        .auth-subtitle {
            text-align: center;
            margin-top: 4px;
            color: #6b7280;
            font-size: 13px;
        }

        .auth-link {
            text-align: center;
            margin-top: 10px;
            color: #9ca3af;
            font-size: 12px;
        }

        .match-ok {
            color: #34d399;
            font-size: 12px;
            margin-top: -6px;
            margin-bottom: 2px;
        }

        .match-bad {
            color: #f87171;
            font-size: 12px;
            margin-top: -6px;
            margin-bottom: 2px;
        }

        .strength {
            font-size: 12px;
            margin-top: -6px;
            margin-bottom: 2px;
            color: #9ca3af;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, "utils", "logo.png")

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    _, card, _ = st.columns([1, 1.2, 1])

    with card:
        with st.container(border=True):
            logo_base64 = get_base64(logo_path)

            st.markdown(
                f"""
                <div style="text-align:center; margin-bottom:18px;">
                    <img src="data:image/png;base64,{logo_base64}" width="140"/>
                    <div class="auth-title">BENCHMARKING PLATFORM</div>
                    <div class="auth-subtitle">Sign in to continue or create an account</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            is_login = st.session_state.auth_mode == "login"
            is_register = st.session_state.auth_mode == "register"

            with col1:
                if st.button(
                    "Sign in",
                    use_container_width=True,
                    type="primary" if is_login else "secondary",
                ):
                    st.session_state.auth_mode = "login"
                    st.rerun()

            with col2:
                if st.button(
                    "Sign Up",
                    use_container_width=True,
                    type="primary" if is_register else "secondary",
                ):
                    st.session_state.auth_mode = "register"
                    st.rerun()

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            username = st.text_input("Username", placeholder="name@neysa.ai").strip()
            password = st.text_input(
                "Password", placeholder="Enter your password", type="password"
            )

            confirm_password = ""
            if st.session_state.auth_mode == "register":
                confirm_password = st.text_input(
                    "Confirm Password",
                    placeholder="Re-enter your password",
                    type="password",
                )

                if password and confirm_password:
                    if password == confirm_password:
                        st.markdown(
                            "<div class='match-ok'>✓ Passwords match</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div class='match-bad'>Passwords do not match</div>",
                            unsafe_allow_html=True,
                        )

                if password:
                    st.markdown(
                        f"<div class='strength'>Password strength: {password_strength(password)}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            valid_username = is_valid_neysa_username(username)

            if username and not valid_username:
                st.error("Username must end with @neysa.ai")

            if st.session_state.auth_mode == "register":
                can_submit = (
                    bool(username)
                    and valid_username
                    and bool(password)
                    and bool(confirm_password)
                    and password == confirm_password
                    and len(password) >= 6
                )

                if st.button(
                    "Create account →",
                    use_container_width=True,
                    type="primary",
                    disabled=not can_submit,
                ):
                    if not valid_username:
                        st.error("Username must end with @neysa.ai")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters long.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        success, msg = register_user(username, password)
                        if success:
                            st.success(msg)
                            st.session_state.auth_mode = "login"
                            st.rerun()
                        else:
                            st.error(msg)

            else:
                can_submit = bool(username) and bool(password) and valid_username

                if st.button(
                    "Sign In",
                    use_container_width=True,
                    type="primary",
                    disabled=not can_submit,
                ):
                    if not valid_username:
                        st.error("Username must be registered with Neysa")
                    else:
                        success, msg = login_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown(
                "<div class='auth-link'>Forgot Password?</div>",
                unsafe_allow_html=True,
            )


