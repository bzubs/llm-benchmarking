import streamlit as st
from auth import register_user, login_user
#auth screen
def show_auth_screen():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #0a0a0f;
    }

    [data-testid="stMainBlockContainer"] {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding-top: 0 !important;
    }

    /* Card container trick via column */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }

    .auth-logo {
        font-family: 'DM Mono', monospace;
        font-size: 28px;
        letter-spacing: 3px;
        color: #6366f1;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .auth-heading {
        font-size: 22px;
        font-weight: 600;
        color: #f1f1f3;
        font-family: 'DM Sans', sans-serif;
        margin-bottom: 2px;
    }
    .auth-sub {
        font-size: 20px;
        color: #4B5563;
        font-family: 'DM Sans', sans-serif;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Center the card using columns
    _, card, _ = st.columns([1, 1.2, 1])

    with card:
        with st.container(border=True):
            st.markdown('<div class="auth-logo">⬡ Benchmarking Platform</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-heading">Welcome back</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-sub">Sign in to your account or create one.</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            menu = st.radio("Auth Mode", ["Login", "Register"], horizontal=True, label_visibility="collapsed")

            username = st.text_input("Username", placeholder="your_username")
            password = st.text_input("Password", placeholder="••••••••", type="password")

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if menu == "Register":
                access_code = st.text_input("Access Code", placeholder="Enter access code", type="password")
                if st.button("Create account →", use_container_width=True, type="primary"):
                    success, msg = register_user(username, password, access_code)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

            if menu == "Login":
                if st.button("Sign in →", use_container_width=True, type="primary"):
                    success, msg = login_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error(msg)

