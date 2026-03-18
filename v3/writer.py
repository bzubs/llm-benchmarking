from datetime import datetime
import os
import json
from schema import BenchTask
import time

def get_log_file(task_id):
    return f"logs/task_{task_id}.log"

def write_benchmark_log(task: BenchTask, duration: float, return_code: int,
                    metrics: dict, stderr: str, stdout: str,):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
{'='*80}
Timestamp: {timestamp}
Model: {task.config.model_name}
GPU ID: {task.gpu_assigned}
Runtime: {duration:.2f} seconds
Return Code: {return_code}

Configuration:
{json.dumps(task.config.model_dump(), indent=2)}

Metrics ({len(metrics)} found):
{json.dumps(metrics, indent=2)}

Stderr Output (last 1000 chars):
{stderr[-1000:] if len(stderr) > 1000 else stderr}

Stdout Output (last 1000 chars):
{stdout[-1000:] if len(stdout) > 1000 else stdout}
{'='*80}
"""
    os.makedirs("summary", exist_ok=True)

    with open(f"summary/{task.id}_summary.log", "a") as f:
        f.write(log_entry)



def append_jsonl_history(task: BenchTask,
                         duration: float,
                         return_code: int,
                         metrics: dict):

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "taskID" : task.id,
        "task_status" : task.status,
        "username" : task.config.username,
        "model": task.config.model_name,
        "gpu": task.gpu_assigned,
        "runtime_sec": duration,
        "return_code": return_code,

        # Core metrics (safe access)
        "successful_requests": metrics.get("successful_requests"),
        "request_throughput": metrics.get("request_throughput"),
        "output_token_throughput": metrics.get("output_token_throughput"),
        "total_token_throughput": metrics.get("total_token_throughput"),

        "median_ttft_ms": metrics.get("median_ttft_ms"),
        "median_tpot_ms": metrics.get("median_tpot_ms"),
        "median_itl_ms": metrics.get("median_itl_ms"),

        # Useful config snapshot
        "num_prompts": task.config.num_prompts,
        "max_concurrency": task.config.max_concurrency,
        "input_len": task.config.input_len,
        "output_len": task.config.output_len,
        "quantization": task.config.quantization,
        "dtype": task.config.dtype,
        "gpu_memory_util": task.config.gpu_memory_util,
        "avg_gpu_util_percent": metrics.get("avg_gpu_util_percent"),
        "peak_gpu_util_percent": metrics.get("peak_gpu_util_percent"),
        "avg_gpu_mem_mb": metrics.get("avg_gpu_mem_mb"),
        "peak_gpu_mem_mb": metrics.get("peak_gpu_mem_mb"),
    }

    with open("runs_history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def write_task_log(task_id, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(get_log_file(task_id), "a") as f:
        f.write(f"[{ts}] {msg}\n")