import os
from writer import write_server_log
import streamlit as st
from huggingface_hub import list_models
from transformers import AutoTokenizer

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
            direction=-1,
            limit=limit,
        )
        return [m.modelId for m in models]
    except Exception:
        return []


@st.cache_data(ttl=3600)
def detect_model_type(model_id: str) -> str:
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        if hasattr(tok, "chat_template") and tok.chat_template:
            return "chat"

    except Exception:
        pass

    return "base"

def tail_file(path, n=300):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = min(8192, f.tell())
        f.seek(-size, os.SEEK_END)
        lines = f.read().decode(errors="ignore").splitlines()
    return "\n".join(lines[-n:])

        
def read_new_logs(filepath, cursor):
    if not os.path.exists(filepath):
        return "", cursor

    with open(filepath, "r") as f:
        f.seek(cursor)
        data = f.read()
        new_cursor = f.tell()

    return data, new_cursor
