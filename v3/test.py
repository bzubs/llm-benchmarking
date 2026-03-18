from cluster import GPUCluster
from scheduler import GPUScheduler
from executor import TaskExecutor
from schema import BenchmarkConfig, BenchTask
import threading
import time


# --- Task factory ---
def create_task(task_id: int) -> BenchTask:
    cfg = BenchmarkConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype="auto",
        max_model_len=8192,
        quantization="fp8",
        num_prompts=10,
        max_concurrency=5
    )

    cfg.dataset_name = "random"
    cfg.dataset_path = None
    cfg.endpoint = "/v1/completions"

    return BenchTask(id=task_id, config=cfg)


# --- Debug printer ---
def print_cluster(cluster):
    print("\n📊 Cluster State:")
    for node in cluster:
        queue_ids = [t.id for t in node.queue]
        statuses = [t.status for t in node.queue]
        print(f"GPU {node.gpu.id} | status={node.gpu.status} | queue={list(zip(queue_ids, statuses))}")


# --- Test ---
def main():
    cluster = GPUCluster([6, 7])  # 👈 ONLY these GPUs
    scheduler = GPUScheduler(cluster)
    executor = TaskExecutor(cluster)

    # Create tasks
    tasks = [create_task(i) for i in range(10)]

    # Schedule tasks
    for task in tasks:
        scheduler.schedule_task(task)
        print(f"Scheduled Task {task.id} → GPU {task.gpu_assigned} | status={task.status}")

    print_cluster(cluster)

    # 🔥 Start executor in background thread
    executor_thread = threading.Thread(target=executor.start, daemon=True)
    executor_thread.start()

    # 🔥 Monitor system
    try:
        while True:
            print_cluster(cluster)

            # Stop condition: all queues empty
            all_empty = all(len(node.queue) == 0 for node in cluster)
            if all_empty:
                print("\n✅ All tasks completed")
                break

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nStopping executor...")

    executor.running = False


if __name__ == "__main__":
    main()