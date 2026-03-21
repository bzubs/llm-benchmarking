import subprocess
import time
import re
import urllib.request
from schema import BenchmarkConfig, BenchTask
from cli_builder import env_for_gpu
from writer import write_task_log, write_benchmark_log, append_jsonl_history
import threading
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    NVMLError
)


def sample_gpu_stats(task_id: int, gpu_id: int, stop_event: threading.Event, samples: list):
    try:
        nvmlInit()

        # IMPORTANT: always index 0 because of CUDA_VISIBLE_DEVICES masking
        handle = nvmlDeviceGetHandleByIndex(0)

        while not stop_event.is_set():
            util = nvmlDeviceGetUtilizationRates(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)

            samples.append({
                "gpu_util_percent": util.gpu,
                "mem_used_mb": int(mem.used) / 1024 / 1024,
                "mem_total_mb": int(mem.total) / 1024 / 1024
            })

            time.sleep(0.2)

    except NVMLError as e:
        write_task_log(task_id, f"[GPU SAMPLER ERROR] {str(e)}")

    finally:
        try:
            nvmlShutdown()
        except:
            pass


def parse_metrics(task_id: int, stdout: str) -> dict:
    metrics = {}

    write_task_log(task_id, "[PARSE_METRICS] Parsing metrics")
    write_task_log(task_id, f"[PARSE_METRICS] Stdout length: {len(stdout)}")

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

    write_task_log(task_id, f"[PARSE_METRICS] Extracted {len(metrics)} metrics")
    return metrics


def wait_for_vllm_ready(host: str, port: int, timeout: int = 180) -> bool:
    start = time.time()
    url = f"http://{host}:{port}/v1/models"

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)

    return False


def start_vllm_server(task_id, cfg: BenchmarkConfig, gpu_id: int, port: int, host: str = "127.0.0.1"):
    cmd = [
        "vllm", "serve", cfg.model_name,
        "--host", host,
        "--port", str(port),
        "--dtype", str(cfg.dtype),
        "--max-model-len", str(cfg.max_model_len),
    ]

    if cfg.quantization != "none":
        cmd += ["--quantization", str(cfg.quantization)]

    env = env_for_gpu(gpu_id)

    write_task_log(task_id, "=" * 80)
    write_task_log(task_id, f"[START SERVER] {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    start_time = time.time()
    timeout = 120

    while True:
        if proc.poll() is not None:
            _, stderr = proc.communicate()

            root_error = None
            for line in reversed(stderr.splitlines()):
                if line.startswith(("ValueError:", "RuntimeError:", "OSError:", "Exception:")):
                    root_error = line
                    break

            if not root_error:
                root_error = "\n".join(stderr.splitlines()[-5:])

            write_task_log(task_id, "[START SERVER] Failed")
            raise RuntimeError(f"vLLM server failed:\n{root_error}")

        if wait_for_vllm_ready(host, port, timeout=2):
            write_task_log(task_id, "[START SERVER] Ready")
            return proc

        if time.time() - start_time > timeout:
            proc.terminate()
            raise RuntimeError("vLLM server startup timed out")

        time.sleep(1)


def serve_then_bench(task: BenchTask, port: int, host: str = "127.0.0.1"):
    cfg = task.config
    gpu_id = task.gpu_assigned
    task_id = task.id


    server_proc = None
    gpu_samples = []
    stop_event = threading.Event()

    try:
        server_proc = start_vllm_server(task_id, cfg, gpu_id, port, host)

        sampler_thread = threading.Thread(
            target=sample_gpu_stats,
            args=(task_id, gpu_id, stop_event, gpu_samples),
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

        write_task_log(task_id, f"[BENCH CMD] {' '.join(cmd)}")

        start = time.time()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = proc.communicate()
        duration = time.time() - start

        stop_event.set()
        sampler_thread.join()

        if proc.returncode != 0:
            write_task_log(task_id, f"[BENCH ERROR] {proc.returncode}")
            task.status = "failed"
            raise RuntimeError("Benchmark failed")

        write_task_log(task_id, "[BENCH DONE]")

        metrics = parse_metrics(task_id, stdout)

        task.status = "completed"

        if gpu_samples:
            metrics.update({
                "avg_gpu_util_percent": sum(s["gpu_util_percent"] for s in gpu_samples) / len(gpu_samples),
                "peak_gpu_util_percent": max(s["gpu_util_percent"] for s in gpu_samples),
                "avg_gpu_mem_mb": sum(s["mem_used_mb"] for s in gpu_samples) / len(gpu_samples),
                "peak_gpu_mem_mb": max(s["mem_used_mb"] for s in gpu_samples),
            })

        write_benchmark_log(task, duration, proc.returncode, metrics, stderr, stdout)
        append_jsonl_history(task, duration, proc.returncode, metrics)

        return {
            "config": cfg.model_dump(),
            "returncode": proc.returncode,
            "runtime_sec": duration,
            "metrics": metrics,
        }

    finally:
        if server_proc:
            write_task_log(task_id, "[CLEANUP] Terminating server")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()