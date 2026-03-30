from fastapi import FastAPI, HTTPException
import requests
from schema import BenchmarkConfig, BenchTaskResponse
import json
import os
from dotenv import load_dotenv
from writer import append_jsonl_history

load_dotenv()

GPU_BACKENDS = json.loads(os.getenv("GPU_BACKENDS", "{}"))


if not GPU_BACKENDS:
    raise ValueError("GPU_BACKENDS is empty.")

app = FastAPI()

task_map = {}  # router_task_id → (backend_url, backend_task_id)
task_counter = 0


@app.post("/submit", response_model=BenchTaskResponse)
def submit(cfg: BenchmarkConfig):
    global task_counter

    gpu_type = cfg.gpu_type

    if gpu_type not in GPU_BACKENDS:
        raise HTTPException(status_code=400, detail="Invalid GPU type")

    backend_url = GPU_BACKENDS[gpu_type]

    try:
        resp = requests.post(f"{backend_url}/submit", json=cfg.model_dump(), timeout=5)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    data = resp.json()
    data = BenchTaskResponse(**data)

    router_task_id = task_counter
    task_counter += 1

    # store mapping
    task_map[router_task_id] = {
        "backend_url": backend_url,
        "backend_task_id": data.id,
        "logged" : False
    }

    resp = BenchTaskResponse(
        id = router_task_id,
        gpu_assigned = data.gpu_assigned,
        status = data.status,
        result = data.result
    )

    return resp

@app.get("/status/{task_id}")
def get_status(task_id: int):
    mapping = task_map.get(task_id)

    if not mapping:
        raise HTTPException(status_code=404, detail="Task not found")

    backend_url = mapping["backend_url"]
    backend_task_id = mapping["backend_task_id"]

    try:
        resp = requests.get(
            f"{backend_url}/status/{backend_task_id}",
            timeout=5
        )
        resp.raise_for_status()

        raw = resp.json()
        print("RAW:", raw)  

        task = BenchTaskResponse(**raw)

        # safe logging
        if task.status in ["completed", "failed"] and not mapping["logged"]:
            try:
                append_jsonl_history(task)
                mapping["logged"] = True
            except Exception as e:
                print("LOGGING ERROR:", e)

        return raw   #return parsed data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")