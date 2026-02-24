from datetime import datetime
import os
import json
from schema import BenchmarkConfig



def write_server_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("server.log", "a") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()        #  force writ
        os.fsync(f.fileno())  # force OS flush



def write_benchmark_log(cfg: BenchmarkConfig, duration: float, return_code: int,
                        metrics: dict, stderr: str, stdout: str,
                        benchmark_mode: str = "Offline"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
{'='*80}
Timestamp: {timestamp}
Benchmark Mode: {benchmark_mode}
Benchmark Type: {cfg.benchmark_type}
Model: {cfg.model_name}
GPU ID: {cfg.device if hasattr(cfg, 'device') else 'N/A'}
Runtime: {duration:.2f} seconds
Return Code: {return_code}

Configuration:
{json.dumps(cfg.model_dump(), indent=2)}

Metrics ({len(metrics)} found):
{json.dumps(metrics, indent=2)}

Stderr Output (last 1000 chars):
{stderr[-1000:] if len(stderr) > 1000 else stderr}

Stdout Output (last 1000 chars):
{stdout[-1000:] if len(stdout) > 1000 else stdout}
{'='*80}
"""
    with open("logs.txt", "a") as f:
        f.write(log_entry)



def append_jsonl_history(cfg: BenchmarkConfig,
                         duration: float,
                         return_code: int,
                         metrics: dict,
                         benchmark_mode: str = "Online"):

    record = {
        "username" : cfg.username,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_mode": benchmark_mode,
        "benchmark_type": cfg.benchmark_type,
        "model": cfg.model_name,
        "gpu": cfg.device if hasattr(cfg, "device") else None,
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
        "num_prompts": cfg.num_prompts,
        "max_concurrency": cfg.max_concurrency,
        "input_len": cfg.input_len,
        "output_len": cfg.output_len,
        "quantization": cfg.quantization,
        "dtype": cfg.dtype,
        "gpu_memory_util": cfg.gpu_memory_util,
        "avg_gpu_util_percent": metrics.get("avg_gpu_util_percent"),
        "peak_gpu_util_percent": metrics.get("peak_gpu_util_percent"),
        "avg_gpu_mem_mb": metrics.get("avg_gpu_mem_mb"),
        "peak_gpu_mem_mb": metrics.get("peak_gpu_mem_mb"),
    }

    with open("runs_history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

