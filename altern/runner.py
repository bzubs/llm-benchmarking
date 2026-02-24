###ALTERNATIVE CODE FOR RUNNER.PY with different loggin structure


import subprocess
import time
import json
import re
import os
import select
from datetime import datetime
from schema import BenchmarkConfig
from cli_builder import env_for_gpu
from writer import write_server_log
from helper import detect_model_type
import urllib.request
import threading

from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    NVMLError
)

#refined version2 with server-logs
# ========================= GPU SAMPLER =========================

def sample_gpu_stats(gpu_id: int, stop_event: threading.Event, samples: list):
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_id)

        while not stop_event.is_set():
            util = nvmlDeviceGetUtilizationRates(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)

            samples.append({
                "gpu_util_percent": util.gpu,
                "mem_used_mb": mem.used / 1024 / 1024,
                "mem_total_mb": mem.total / 1024 / 1024
            })

            time.sleep(0.2)

    except NVMLError as e:
        write_server_log(f"[GPU SAMPLER ERROR] {str(e)}")

    finally:
        try:
            nvmlShutdown()
        except:
            pass


# ========================= METRICS PARSING =========================

def parse_metrics(stdout: str, benchmark_type: str) -> dict:
    metrics = {}

    write_server_log(f"[PARSE_METRICS] Parsing metrics for {benchmark_type}")

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
                value = m.group(1)
                metrics[key] = float(value) if "." in value else int(value)

    write_server_log(f"[PARSE_METRICS] Extracted {len(metrics)} metrics")
    return metrics


# ========================= ERROR CHECK =========================
def check_metrics_parsing_error(metrics: dict,
                                stdout: str,
                                stderr: str,
                                benchmark_type: str,
                                return_code: int):

    # 1️⃣ Process-level failure
    if return_code != 0:
        return True, f"Benchmark process exited with code {return_code}"

    # 2️⃣ Connection issues
    if "Connect call failed" in stdout or "ConnectionRefusedError" in stdout:
        return True, "Failed to connect to server."

    # 3️⃣ Serve-specific validation
    if benchmark_type == "serve":
        successful = metrics.get("successful_requests", 0)

        if successful == 0:
            return True, "Benchmark completed but zero successful requests."

    # 4️⃣ No metrics parsed at all
    if not metrics:
        return True, "No metrics were parsed from benchmark output."

    return False, ""



# ========================= READINESS CHECK =========================

def wait_for_vllm_ready(host: str, port: int, timeout: int = 180) -> bool:
    start = time.time()
    url = f"http://{host}:{port}/v1/models"

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except:
            pass
        time.sleep(1)

    return False


# ========================= SERVER START =========================

def start_vllm_server(cfg: BenchmarkConfig,
                      gpu_id: int,
                      host: str,
                      port: int,
                      timeout: int = 180):

    cmd = [
        "vllm", "serve", cfg.model_name,
        "--host", host,
        "--port", str(port),
        "--dtype", str(cfg.dtype),
        "--max-model-len", str(cfg.max_model_len)
    ]

    if cfg.quantization != "none":
        cmd += ["--quantization", str(cfg.quantization)]

    env = env_for_gpu(gpu_id)

    write_server_log("=" * 80)
    write_server_log(f"[START SERVER] {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    start_time = time.time()
    logs_buffer = []

    while time.time() - start_time < timeout:

        # Detect crash
        if proc.poll() is not None:
            remaining = proc.stdout.read() if proc.stdout else ""
            logs_buffer.append(remaining)

            raise RuntimeError(
                f"vLLM crashed during startup.\n"
                f"Exit code: {proc.returncode}\n"
                f"Logs:\n{''.join(logs_buffer)[-2000:]}"
            )

        # Non-blocking stdout read
        if proc.stdout:
            ready, _, _ = select.select([proc.stdout], [], [], 0.1)
            if ready:
                line = proc.stdout.readline()
                if line:
                    logs_buffer.append(line)
                    write_server_log(f"[SERVER] {line.strip()}")

                    if "CUDA out of memory" in line:
                        raise RuntimeError("CUDA OOM during model load.")

                    if "Address already in use" in line:
                        raise RuntimeError("Port already in use.")

        # Check readiness
        if wait_for_vllm_ready(host, port, timeout=1):
            write_server_log("[READY] vLLM server is ready")
            return proc

        time.sleep(0.5)

    proc.terminate()
    raise RuntimeError("vLLM server did not become ready within timeout.")


# ========================= SERVE THEN BENCH =========================

def serve_then_bench(cfg: BenchmarkConfig, gpu_id: int = 0,
                     host: str = "127.0.0.1", port: int = 8000):

    server_proc = None
    gpu_samples = []
    stop_event = threading.Event()

    try:
        server_proc = start_vllm_server(cfg, gpu_id, host, port)

        sampler_thread = threading.Thread(
            target=sample_gpu_stats,
            args=(gpu_id, stop_event, gpu_samples),
            daemon=True
        )
        sampler_thread.start()

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

        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start

        stop_event.set()
        sampler_thread.join()

        metrics = parse_metrics(proc.stdout, cfg.benchmark_type)

        has_error, error_msg = check_metrics_parsing_error(
            metrics,
            proc.stdout,
            proc.stderr,
            cfg.benchmark_type,
            proc.returncode
        )

        if has_error:
            write_server_log(f"[BENCH ERROR] {error_msg}")

        # GPU aggregation
        if gpu_samples:
            metrics.update({
                "avg_gpu_util_percent":
                    sum(s["gpu_util_percent"] for s in gpu_samples) / len(gpu_samples),
                "peak_gpu_util_percent":
                    max(s["gpu_util_percent"] for s in gpu_samples),
                "avg_gpu_mem_mb":
                    sum(s["mem_used_mb"] for s in gpu_samples) / len(gpu_samples),
                "peak_gpu_mem_mb":
                    max(s["mem_used_mb"] for s in gpu_samples),
            })

        return {
            "config": cfg.model_dump(),
            "returncode": proc.returncode,
            "runtime_sec": duration,
            "metrics": metrics,
            "stderr": proc.stderr,
            "stdout": proc.stdout,
            "error": error_msg if has_error else None,
        }

    finally:
        if server_proc:
            write_server_log("[CLEANUP] Terminating vLLM server")
            server_proc.terminate()
            server_proc.wait(timeout=10)
            write_server_log("=" * 80)


# ========================= MAIN =========================

def main():
    cfg = BenchmarkConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype="auto",
        max_model_len=8192,
        quantization="fp8",
        num_prompts=10,
        max_concurrency=5,
        benchmark_type="serve",
        backend="vllm"
    )

    model_type = detect_model_type(cfg.model_name)

    if model_type == "chat":
        cfg.endpoint = "/v1/chat/completions"
    else:
        cfg.endpoint = "/v1/completions"

    serve_then_bench(cfg)


if __name__ == "__main__":
    main()
