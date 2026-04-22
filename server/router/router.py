import requests
import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends
from datetime import datetime

from schemas.task import BenchmarkConfig, BenchTask
from schemas.user import User
from router.writer import append_jsonl_history
from config_loader import load_config
from router.verification import get_current_user
from router.auth import create_access_token

from database.database import DataBase
from database.db_service import DBService

load_dotenv()

# loading configs from config.yaml
config = load_config()

gpu_backends = config.gpu_backends
history_file = config.history_file

if not gpu_backends:
    raise ValueError("gpu_backends in config file is empty.")

app = FastAPI()

# DB init
DB_URL = os.getenv("DB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "bench_db")

db = DataBase(url=DB_URL)
db.connect(DB_NAME)

db_service = DBService(db)


# Auth


@app.post("/register")
def register(user: User):
    success, msg = db_service.register_user(
        user.username, user.password)

    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {"message": msg}


@app.post("/login")
def login(user: User):
    success, msg = db_service.login_user(user.username, user.password)

    if not success:
        raise HTTPException(status_code=401, detail=msg)

    token = create_access_token({"sub": user.username})

    return {"message": msg, "access_token": token, "token_type": "bearer"}


# Task-post
@app.post("/submit", response_model=BenchTask)
def submit(cfg: BenchmarkConfig, username: str = Depends(get_current_user)):
    # validate user
    user = db.get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    gpu_type = cfg.gpu_type

    if gpu_type not in gpu_backends:
        raise HTTPException(status_code=400, detail="Invalid GPU type")

    backend_url = gpu_backends[gpu_type]

    try:
        resp = requests.post(f"{backend_url}/submit", json=cfg.model_dump(), timeout=5)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    data = resp.json()
    if "id" in data:
        data["id"] = str(data["id"])
    data = BenchTask(**data)

    now = datetime.utcnow()

    task = BenchTask(
        id="",
        username=username,
        config=data.config,
        gpu_assigned=data.gpu_assigned,
        status=data.status,
        result=data.result,
        created_at=now,
        updated_at=now,
    )

    success, task_id, msg = db_service.create_task(
        task,
        backend_url=backend_url,
        backend_task_id=data.id,
        logged=False,
    )

    if not success:
        raise HTTPException(status_code=500, detail=msg)

    task.id = task_id
    return task


# Status


@app.get("/status/{task_id}")
def get_status(task_id: str):
    task_doc = db_service.get_task(task_id)

    if not task_doc:
        raise HTTPException(status_code=404, detail="Task not found")

    backend_url = task_doc.get("backend_url")
    backend_task_id = int(task_doc.get("backend_task_id"))

    if not backend_url or backend_task_id is None:
        raise HTTPException(status_code=500, detail="Backend mapping missing for task")

    try:
        resp = requests.get(f"{backend_url}/status/{backend_task_id}", timeout=5)
        resp.raise_for_status()

        raw = resp.json()
        if "id" in raw:
            raw["id"] = str(raw["id"])

        backend_task = BenchTask(**raw)

        # DB UPDATE
        if backend_task.result is not None:
            result_data = (
                backend_task.result.model_dump()
                if hasattr(backend_task.result, "model_dump")
                else backend_task.result
            )

            success, msg = db_service.update_task_status(
                task_id,
                backend_task.status,
                result_data,
                datetime.utcnow(),
            )
            if not success:
                raise HTTPException(status_code=500, detail=msg)

        router_task = BenchTask(
            id=task_id,
            username=backend_task.username or task_doc["username"],
            config=backend_task.config,
            gpu_assigned=backend_task.gpu_assigned,
            status=backend_task.status,
            result=backend_task.result,
        )

        # existing logging
        if backend_task.status in ["completed", "failed"] and not task_doc.get(
            "logged", False
        ):
            try:
                append_jsonl_history(history_file, router_task)
                success, msg = db_service.mark_task_logged(task_id)
                if not success:
                    raise RuntimeError(msg)

            except Exception as e:
                raise RuntimeError(f"LOGGING ERROR: {e}")

        return router_task

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")


# HISTORY


@app.get("/history")
def get_history(username: str = Depends(get_current_user)):
    user = db.get_user(username)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    tasks = db_service.get_user_tasks(username)

    return {"tasks": tasks}
