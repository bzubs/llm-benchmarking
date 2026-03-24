import threading
import time
from runner import serve_then_bench
from schema import BenchResult
from writer import append_jsonl_history

class TaskExecutor:
    def __init__(self, cluster):
        self.cluster = cluster
        self.running = True

    def start(self):
        while self.running:
            for node in self.cluster:

                if not node.queue:
                    continue

                task = node.queue[0]

                if task.status != "assigned":
                    continue

                # mark as running BEFORE thread starts (avoid double scheduling)
                task.status = "running"

                t = threading.Thread(
                    target=self.run_task,
                    args=(node, task),
                    daemon=True
                )
                t.start()

            time.sleep(0.5)  # avoid tight loop

    def run_task(self, node, task):
        gpu_id = node.gpu.id
        port = 8000 + gpu_id  # unique port per GPU

        task.config.port = str(port)

        print(f"[EXECUTOR] Running Task {task.id} on GPU {gpu_id}")

        result = {}

        try:
            result = serve_then_bench(task, port=port)

            if result["returncode"] == 0:
                print(f"[EXECUTOR] Completed Task {task.id}")
                result = BenchResult(**result)
                task.result = result
                task.status = "completed"

            else:
                print(f"[EXECUTOR] Failed Task {task.id}")
                task.status = "failed"

        except Exception as e:
            print(f"[EXECUTOR] Error Task {task.id}: {e}")
            task.status = "failed"
            task.result = BenchResult(
                config=task.config.model_dump(),
                returncode=-1,
                runtime_sec=0,
                metrics={"error": str(e)}
            )


        runtime = 0.0
        code = 4
        metrics = {}

        if isinstance(result, BenchResult):
            runtime = result.runtime_sec or 0.0
            code = result.returncode
            metrics = result.metrics
        elif isinstance(result, dict):
            runtime = result.get("runtime_sec") or 0.0
            code = result.get("returncode") or -1
            metrics = result.get("metrics") or {}

        append_jsonl_history(task, runtime, code, metrics)    


        # CRITICAL SECTION
        self.cluster.pop_task(gpu_id, task)

        if node.queue:
            next_task = node.queue[0]
            next_task.status = "assigned"
        else:
            node.gpu.status = "free"

        