import threading
import time
import requests
from runner import serve_then_bench
from schema import BenchResult
from writer import append_jsonl_history


class TaskExecutor:
    def __init__(self, cluster, scheduler):
        self.cluster = cluster
        self.running = True
        self.scheduler = scheduler

    def start(self):
        while self.running:

            for task in list(self.cluster.pending_tasks):

                # skip non-assigned
                if task.status != "assigned":
                    continue

                # prevent double scheduling
                task.status = "running"

                t = threading.Thread(
                    target=self.run_task,
                    args=(task,),
                    daemon=True
                )
                t.start()

            time.sleep(0.5)

    def run_task(self, task):
        gpu_ids = task.gpu_assigned  
        primary_gpu = gpu_ids[0]     # pick one for port mapping

        port = 8000 + primary_gpu
        task.config.port = str(port)

        print(f"[EXECUTOR] Running Task {task.id} on GPUs {gpu_ids}")

        try:
            # serve_then_bench already uses task.gpu_assigned internally
            serve_then_bench(task, port=port)

            if task.status == "completed":
                print(f"[EXECUTOR] Completed Task {task.id}")
            else:
                print(f"[EXECUTOR] Failed Task {task.id}")

        except Exception as e:
            print(f"[EXECUTOR] Error Task {task.id}: {e}")

            task.status = "failed"
            task.result = BenchResult(
                config=task.config,
                returncode=-1,
                runtime_sec=0,
                metrics={},
                error_msg=str(e)
            )

        #persist
        append_jsonl_history(
            task,
            task.result.runtime_sec if task.result else 0,
            task.result.returncode if task.result else -1,
            task.result.metrics if task.result else {}
        )

        requests.post(
            f"http://{ROUTER_IP}:{PORT}/log",
            json={
                "task": task.model_dump(),
                "runtime_sec": task.result.runtime_sec if task.result else 0,
                "returncode": task.result.returncode if task.result else -1,
                "metrics": task.result.metrics if task.result else {}
            }
        )

        # release all GPUs
        self.cluster.release_gpus(gpu_ids)

        self.scheduler.try_schedule_pending_tasks()

        # remove from global queue
        if task in self.cluster.pending_tasks:
            self.cluster.pending_tasks.remove(task)