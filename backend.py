from fastapi import FastAPI, HTTPException
from schema import BenchmarkConfig, BenchTask, BenchTaskResponse
from scheduler import GPUScheduler
from cluster import GPUCluster
from executor import TaskExecutor
from dotenv import load_dotenv
import os
import threading

load_dotenv()
app = FastAPI()

gpu_ids_env = os.getenv("GPU_IDS", "")
gpu_ids_env = gpu_ids_env.strip("[]")
gpu_id_list = [int(x.strip()) for x in gpu_ids_env.split(",") if x.strip()]

cluster = GPUCluster(gpu_id_list)
scheduler = GPUScheduler(cluster)

task_counter = 0
tasks = {}
counter_lock = threading.Lock()


@app.post("/submit", response_model=BenchTaskResponse)
def submit_task(cfg: BenchmarkConfig):
    global task_counter

    # validation
    if cfg.n_gpus_required > cluster.get_cluster_size():
        raise HTTPException(
            status_code=400,
            detail="Requested GPUs exceed available GPUs"
        )

    with counter_lock:
        task_id = task_counter
        task_counter += 1

    task = BenchTask(
        id=task_id,
        config=cfg,
        status="init"
    )

    tasks[task.id] = task

    scheduler.schedule_task(task)

    return BenchTaskResponse(
        id=task.id,
        gpu_assigned=task.gpu_assigned,  # now List[int] or None
        status=task.status
    )


@app.get("/status/{task_id}", response_model=BenchTaskResponse)
def get_status(task_id: int):
    task = tasks.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return BenchTaskResponse(
        id=task.id,
        gpu_assigned=task.gpu_assigned,
        status=task.status,
        result=task.result
    )