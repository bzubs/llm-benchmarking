import threading
import uvicorn
import os
from backend import app, cluster
from executor import TaskExecutor
from dotenv import load_dotenv

load_dotenv()

backend_port = os.getenv("BACKEND_PORT")
# Create executor
executor = TaskExecutor(cluster)

def start_executor():
    executor.start()

if __name__ == "__main__":
    # Run executor in background
    t = threading.Thread(target=start_executor, daemon=True)
    t.start()

    print("[MAIN] Executor started")

    # Start FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8023)