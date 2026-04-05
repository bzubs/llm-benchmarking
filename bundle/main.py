import threading
import uvicorn
import os
from backend import app, cluster, scheduler
from executor import TaskExecutor
from config_loader import load_config


"""Handles threading of executor and backend of the GPU; so executor and 
backend run parallely"""


config = load_config()

backend_port = config.backend_port
# Create executor
executor = TaskExecutor(cluster, scheduler)

def start_executor():
    executor.start()

if __name__ == "__main__":
    # Run executor in background
    t = threading.Thread(target=start_executor, daemon=True)
    t.start()

    print("[MAIN] Executor started")

    # Start FastAPI
    uvicorn.run(app, host="0.0.0.0", port=backend_port)

