import subprocess
import time
import json
import re
import os
from datetime import datetime
from schema import BenchmarkConfig
from cli_builder import env_for_gpu
from writer import write_server_log
from helper import detect_model_type
import urllib.request


def append_jsonl_history(cfg: BenchmarkConfig,
                         duration: float,
                         return_code: int,
                         metrics: dict,
                         benchmark_mode: str = "Offline"):

    record = {
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
    }

    with open("runs_history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


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


def check_metrics_parsing_error(metrics: dict, stderr: str,
                                stdout: str, benchmark_type: str):
    if "All requests failed" in stderr:
        return True, "All requests failed during benchmark."

    if "Connect call failed" in stderr or "ConnectionRefusedError" in stderr:
        return True, "Failed to connect to server."

    if benchmark_type == "serve":
        zero_metrics = [
            "successful_requests",
            "request_throughput",
            "output_token_throughput",
            "total_token_throughput",
        ]
        if metrics and all(metrics.get(m, 0) == 0 for m in zero_metrics):
            return True, "All metrics are zero. No requests succeeded."

    return False, ""


def parse_metrics(stdout: str, benchmark_type: str) -> dict:
    metrics = {}

    write_server_log(f"[PARSE_METRICS] Parsing metrics for {benchmark_type}")
    write_server_log(f"[PARSE_METRICS] Stdout length: {len(stdout)}")

    if benchmark_type == "serve":
        patterns = {
            "successful_requests": r"Successful requests:\s+(\d+)",
            "benchmark_duration_sec": r"Benchmark duration \(s\):\s+([\d.]+)",
            "total_input_tokens": r"Total input tokens:\s+(\d+)",
            "total_generated_tokens": r"Total generated tokens:\s+(\d+)",
            "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
            "output_token_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
            "total_token_throughput": r"Total token throughput \(tok/s\):\s+([\d.]+)",
            "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
            "median_ttft_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
            "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
            "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
            "median_tpot_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
            "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
            "mean_itl_ms": r"Mean ITL \(ms\):\s+([\d.]+)",
            "median_itl_ms": r"Median ITL \(ms\):\s+([\d.]+)",
            "p99_itl_ms": r"P99 ITL \(ms\):\s+([\d.]+)",
        }

        for key, pattern in patterns.items():
            m = re.search(pattern, stdout)
            if m:
                metrics[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))

    write_server_log(f"[PARSE_METRICS] Extracted {len(metrics)} metrics")
    return metrics


#Checking for server readiness
def wait_for_vllm_ready(host: str, port: int, timeout: int = 180) -> bool:
    start = time.time()
    url = f"http://{host}:{port}/v1/models"

    write_server_log(f"[READY] Waiting for /v1/models (timeout={timeout}s)")

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    write_server_log("[READY] vLLM server is ready")
                    return True
        except Exception:
            pass
        time.sleep(1)

    write_server_log("[READY] Timeout waiting for vLLM readiness")
    return False


def start_vllm_server(cfg: BenchmarkConfig, gpu_id: int = 7,
                      host: str = "127.0.0.1", port: int = 8000):

    cmd = [
        "vllm", "serve", cfg.model_name,
        "--host", host,
        "--port", str(port),
        "--dtype", str(cfg.dtype),
        "--max-model-len", str(cfg.max_model_len),
        
    ]

    if cfg.quantization!= "none":
        cmd += ["--quantization", str(cfg.quantization)]


    env = env_for_gpu(gpu_id)
    write_server_log(f"="*80)
    write_server_log(f"[START SERVER] Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if not wait_for_vllm_ready(host, port):
        proc.terminate()
        raise RuntimeError("vLLM server did not become ready")

    return proc



def serve_then_bench(cfg: BenchmarkConfig, gpu_id: int = 7,
                     host: str = "127.0.0.1", port: int = 8000):

    server_proc = None
    try:
        server_proc = start_vllm_server(cfg, gpu_id, host, port)

        cmd = [
            "vllm", "bench", "serve",
            "--backend", cfg.backend,
            "--model", cfg.model_name,
            "--endpoint", cfg.endpoint,
            "--host", host,
            "--port", str(port),
        ]

        if cfg.num_prompts:
            cmd += ["--num-prompts", str(cfg.num_prompts)]
        if cfg.input_len:
            cmd += ["--random-input-len", str(cfg.input_len)]
        if cfg.output_len:
            cmd += ["--random-output-len", str(cfg.output_len)]
        if cfg.max_concurrency:
            cmd += ["--max-concurrency", str(cfg.max_concurrency)]

        write_server_log(f"[SERVE_THEN_BENCH] {' '.join(cmd)}")

        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start

        metrics = parse_metrics(proc.stdout, cfg.benchmark_type)

        write_benchmark_log(
            cfg, duration, proc.returncode,
            metrics, proc.stderr, proc.stdout,
            benchmark_mode="Online",
        )

        append_jsonl_history(
            cfg, duration, proc.returncode,
            metrics,
            benchmark_mode="Online",
        )


        return {
            "config": cfg.model_dump(),
            "returncode": proc.returncode,
            "runtime_sec": duration,
            "metrics": metrics,
            "stderr": proc.stderr,
            "stdout": proc.stdout,
        }

    finally:
        if server_proc:
            write_server_log("[SERVE_THEN_BENCH] Terminating server")
            write_server_log("[SERVE_THEN_BENCH] Benchmark Completed")
            server_proc.terminate()
            write_server_log(f"="*80)


def main():
    cfg = BenchmarkConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype="auto",
        max_model_len=8192,
        quantization="fp8",
        num_prompts=10,
        max_concurrency=5
    )

    model_type = detect_model_type(cfg.model_name)

    if model_type == "chat":
        cfg.dataset_name = "sharegpt"
        cfg.dataset_path = "ShareGPT_V3_unfiltered_cleaned_split.json"
        cfg.endpoint = "/v1/chat/completions"
    else:
        cfg.dataset_name = "random"
        cfg.dataset_path = None
        cfg.endpoint = "/v1/completions"

    serve_then_bench(cfg)


if __name__ == "__main__":
    main()
