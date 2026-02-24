from datetime import datetime
import os
def write_server_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("server.log", "a") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()        #  force writ
        os.fsync(f.fileno())  # force OS flush


