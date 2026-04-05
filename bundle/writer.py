from datetime import datetime
import os
import json
from schema import BenchTaskResponse, BenchTask
import time



def write_benchmark_log(task: BenchTask, duration: float, return_code: int,
                    metrics: dict, stderr: str, stdout: str,):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
    {'='*180}
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
    {'='*180}
    """

    with open(f"summary.log", "a") as f:
        f.write(log_entry)

    return log_entry



def append_jsonl_history(response: BenchTaskResponse):
    if response and response.result:
        metrics = response.result.metrics
        if metrics:

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "taskID": response.id,
                "task_status": response.status,
                "username": response.result.config.username,
                "model": response.result.config.model_name,
                "gpu": response.gpu_assigned,
                "runtime_sec": response.result.runtime_sec,
                "returncode": response.result.returncode,
                # Core metrics (safe access)
                "successful_requests": metrics.get("successful_requests"),
                "request_throughput": metrics.get("request_throughput"),
                "output_token_throughput": metrics.get("output_token_throughput"),
                "total_token_throughput": metrics.get("total_token_throughput"),
                "median_ttft_ms": metrics.get("median_ttft_ms"),
                "median_tpot_ms": metrics.get("median_tpot_ms"),
                "median_itl_ms": metrics.get("median_itl_ms"),
                # Useful config snapshot
                "num_prompts": response.result.config.num_prompts,
                "max_concurrency": response.result.config.max_concurrency,
                "input_len": response.result.config.input_len,
                "output_len": response.result.config.output_len,
                "quantization": response.result.config.quantization,
                "dtype": response.result.config.dtype,
                "gpu_memory_util": response.result.config.gpu_memory_util,
                "avg_gpu_util_percent": metrics.get("avg_gpu_util_percent"),
                "peak_gpu_util_percent": metrics.get("peak_gpu_util_percent"),
                "avg_gpu_mem_mb": metrics.get("avg_gpu_mem_mb"),
                "peak_gpu_mem_mb": metrics.get("peak_gpu_mem_mb"),
            }

        else:
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "taskID": response.id,
                "task_status": response.status,
                "username": response.result.config.username,
                "model": response.result.config.model_name,
                "gpu": response.gpu_assigned,
                "runtime_sec": response.result.runtime_sec,
                "returncode": response.result.returncode,
                # Useful config snapshot
                "num_prompts": response.result.config.num_prompts,
                "max_concurrency": response.result.config.max_concurrency,
                "input_len": response.result.config.input_len,
                "output_len": response.result.config.output_len,
                "quantization": response.result.config.quantization,
                "dtype": response.result.config.dtype,
                "gpu_memory_util": response.result.config.gpu_memory_util,
            }

        with open("runs_history.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")


def get_last_log_block(log_file: str) -> str:
    with open(log_file, "r") as f:
        content = f.read()

    blocks = content.split("=" * 80)

    # remove empty blocks
    blocks = [b.strip() for b in blocks if b.strip()]

    if not blocks:
        return ""

    return blocks[-1]


def write_task_log(task_id, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("process.log", "a") as f:
        f.write(f"[{ts}] {msg}\n")

