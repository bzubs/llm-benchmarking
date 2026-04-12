import os
import streamlit as st
import requests
from huggingface_hub import list_models
from transformers import AutoTokenizer
from config_loader import load_config


config = load_config()

backend_url = f"http://localhost:{config.router_port}"


@st.cache_data(ttl=3600)
def fetch_hf_models(limit: int = 200):
    """
    Fetch public text-generation models from Hugging Face.
    Cached for 1 hour.
    """
    try:
        models = list_models(
            filter=["text-generation"],
            sort="downloads",
            limit=limit,
        )
        return [m.id for m in models]
    except Exception:
        return []


@st.cache_data(ttl=3600)
def detect_model_type(model_id: str) -> str:
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if hasattr(tok, "chat_template") and tok.chat_template:
            return "chat"

    except Exception:
        pass

    return "base"


def get_auth_headers():
    token = st.session_state.get("token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def fetch_history():
    try:
        res = requests.get(f"{backend_url}/history", headers=get_auth_headers())

        if res.status_code != 200:
            return None, res.text

        return res.json().get("tasks", []), None

    except Exception as e:
        return None, str(e)

# ---- metrics extraction ----
def get_metric(row, key):
    try:
        return row["result"]["metrics"].get(key)
    except:
        return None
