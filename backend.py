from fastapi import FastAPI, HTTPException
from schema import BenchmarkConfig, BenchTask, BenchTaskResponse
from scheduler import GPUScheduler
from cluster import GPUCluster

app = FastAPI()

cluster = GPUCluster([6, 7])
scheduler = GPUScheduler(cluster)

task_counter = 0
tasks = {}


@app.post("/submit", response_model=BenchTaskResponse)
def submit_task(cfg: BenchmarkConfig):
    global task_counter

    task = BenchTask(
        id=task_counter,
        config=cfg,
        status = "init"
    )

    tasks[task.id] = task

    scheduler.schedule_task(task)

    task_counter += 1

    return BenchTaskResponse(
        id=task.id,
        gpu_assigned=task.gpu_assigned,
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
        result = task.result
    )