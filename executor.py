import threading
import time
from runner import serve_then_bench

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
        port = 8000 + gpu_id  #unique port per GPU

        print(f"[EXECUTOR] Running Task {task.id} on GPU {gpu_id}")

        try:
            result = serve_then_bench(task, port=port)

            if result["returncode"] == 0:
                print(f"[EXECUTOR] Completed Task {task.id}")
                task.status = "completed"

            else:
                print(f"[EXECUTOR] Failed Task {task.id}")
                task.status = "failed"

        except Exception as e:
            print(f"[EXECUTOR] Error Task {task.id}: {e}")
            task.status = "failed"

        # CRITICAL SECTION
        self.cluster.pop_task(gpu_id, task)

        if node.queue:
            next_task = node.queue[0]
            next_task.status = "assigned"
        else:
            node.gpu.status = "free"