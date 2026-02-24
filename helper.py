import os
from writer import write_server_log



def detect_model_type(model_id: str) -> str:
    try:
        write_server_log("[LOADING TOK] Tokenizer being loaded to check model type")
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        write_server_log("[TOK LOADED] Successfully Loaded Tokenizer")

        if hasattr(tok, "chat_template") and tok.chat_template:
            return "chat"

    except Exception:
        pass

    return "base"


def tail_file(path, n=300):
    if not os.path.exists(path):
        return "Waiting for logs..."

    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
        return "".join(lines[-n:])


        
def read_new_logs(filepath, cursor):
    if not os.path.exists(filepath):
        return "", cursor

    with open(filepath, "r") as f:
        f.seek(cursor)
        data = f.read()
        new_cursor = f.tell()

    return data, new_cursor
