import subprocess
import time
import socket
import re
import urllib.request
import os
from typing import List

from schema import BenchmarkConfig, BenchTask, BenchResult
from writer import write_task_log, write_benchmark_log, get_last_log_block
from env_setter import env_for_gpu

import threading

from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    NVMLError
)



def is_port_in_use(host, port):
    """checks if current port on that host is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0



def sample_gpu_stats(task_id: int, gpu_ids: list[int], stop_event: threading.Event, samples: list):
    """starts a thread to monitor GPU utilization during the benchmark run"""
    try:
        nvmlInit()

        handles = {
            gpu_id: nvmlDeviceGetHandleByIndex(gpu_id)
            for gpu_id in gpu_ids
        }

        while not stop_event.is_set():
            snapshot = {
                "timestamp": time.time(),
                "gpus": {}
            }

            for gpu_id, handle in handles.items():
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)

                snapshot["gpus"][str(gpu_id)] = {
                    "gpu_util_percent": util.gpu,
                    "mem_used_mb": mem.used / 1024 / 1024,
                    "mem_total_mb": mem.total / 1024 / 1024,
                }

            samples.append(snapshot)
            time.sleep(0.05)

    except NVMLError as e:
        write_task_log(task_id, f"[GPU SAMPLER ERROR] {str(e)}")

    finally:
        try:
            nvmlShutdown()
        except:
            pass



def parse_metrics(task_id: int, stdout: str) -> dict:
    """parses STDOUT of the current process; to extract metrics using regex and returns a dict. 
    Flexible schema of metrics so any change in metrics structure can be easily updated"""

    metrics = {}
    try:  
        patterns = {
            "successful_requests": r"Successful requests:\s*(\d+)",
            "benchmark_duration_sec": r"Benchmark duration \(s\):\s*([0-9.+-eE]+)",
            "total_input_tokens": r"Total input tokens:\s*(\d+)",
            "total_generated_tokens": r"Total generated tokens:\s*(\d+)",
            "request_throughput": r"Request throughput \(req/s\):\s*([0-9.+-eE]+)",
            "total_token_throughput": r"Total token throughput \(tok/s\):\s*([0-9.+-eE]+)",

            "median_ttft_ms": r"Median TTFT \(ms\):\s*([0-9.+-eE]+)",
            "p99_ttft_ms": r"P99 TTFT \(ms\):\s*([0-9.+-eE]+)",
            "p95_ttft_ms": r"P95 TTFT \(ms\):\s*([0-9.+-eE]+)",

            "median_tpot_ms": r"Median TPOT \(ms\):\s*([0-9.+-eE]+)",
            "p99_tpot_ms": r"P99 TPOT \(ms\):\s*([0-9.+-eE]+)",
            "p95_tpot_ms": r"P95 TPOT \(ms\):\s*([0-9.+-eE]+)",

            "median_itl_ms": r"Median ITL \(ms\):\s*([0-9.+-eE]+)",
            "p99_itl_ms": r"P99 ITL \(ms\):\s*([0-9.+-eE]+)",
            "p95_itl_ms": r"P95 ITL \(ms\):\s*([0-9.+-eE]+)",

            "median_e2el_ms": r"Median E2EL \(ms\):\s*([0-9.+-eE]+)",
            "p99_e2el_ms": r"P99 E2EL \(ms\):\s*([0-9.+-eE]+)",
            "p95_e2el_ms": r"P95 E2EL \(ms\):\s*([0-9.+-eE]+)",
        }

        for key, pattern in patterns.items():
            m = re.search(pattern, stdout)
            if m:
                metrics[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))

    except Exception as e:
        write_task_log(task_id, "[PARSE METRICS] Error Parsing " + str(e))
        raise RuntimeError(e)

    if len(metrics.keys()) <= 0:
        write_task_log(task_id, "[PARSE METRICS] Parsed Nothing. Likely Failure")  
    else:
        write_task_log(task_id, "[PARSE METRICS] Parsing Successful")

    return metrics



def wait_for_vllm_ready(host: str, port: int, timeout: int = 180) -> bool:
    """checks if server is ready by continously polling endpoint v1/models of the server at given host and port"""
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



def start_vllm_server(task: BenchTask, gpu_ids: list[int], port: int, host: str = "127.0.0.1"):
    """Starts a vLLM server with provided config in BenchTask on provided GPUs and port"""
    task_id = task.id
    cfg = task.config

    #(NEW) v6+: Port conflict detection
    if is_port_in_use(host, port):
        error_msg = f"Port {port} already in use"

        result = BenchResult(
            config=cfg,
            returncode=-1,
            runtime_sec=0,
            metrics={},
            error_msg=error_msg
        )

        task.result = result
        task.status = "failed"

        write_task_log(task_id, "[START SERVER] " + error_msg)
        raise RuntimeError(error_msg)

    cmd = [
        "vllm", "serve", cfg.model_name,
        "--host", host,
        "--port", str(port),
        "--dtype", cfg.dtype.value,
        "--max-model-len", str(cfg.max_model_len),
    ]

    if cfg.quantization != "none":
        cmd += ["--quantization", cfg.quantization.value]
    if cfg.tp_size > 1:
        cmd += ["--tensor-parallel-size", str(cfg.tp_size)]
    if cfg.dp_size > 1:
        cmd += ["--data-parallel-size", str(cfg.dp_size)]

    env = env_for_gpu(gpu_ids)

    write_task_log(task_id, "=" * 80)
    write_task_log(task_id, f"[START SERVER] CMD FORMED {' '.join(cmd)}")
    write_task_log(task_id, f"[GPU CONFIG] USING GPUs: {gpu_ids}")

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
            lines = stderr.splitlines()

            # Look for OOM / CUDA errors FIRST (highest priority)
            for line in reversed(lines):
                l = line.lower()
                if "out of memory" in l or "cuda" in l:
                    root_error = line
                    break

            # fallback to Python exceptions
            if not root_error:
                for line in reversed(lines):
                    if line.startswith(("ValueError:", "RuntimeError:", "OSError:", "Exception:")):
                        root_error = line
                        break

            # final fallback
            if not root_error:
                root_error = "\n".join(lines[-10:])  # slightly more context

            error_msg = f"vLLM server startup failed:\n{root_error}"

            result = BenchResult(
                config=cfg,
                returncode=-1,
                runtime_sec=0,
                metrics={},
                error_msg=error_msg
            )

            task.result = result
            task.status = "failed"

            write_task_log(task_id, "[START SERVER]" + error_msg)
            raise RuntimeError(error_msg)

        if wait_for_vllm_ready(host, port, timeout=2):
            write_task_log(task_id, "[START SERVER] vLLM Server Ready")
            return proc

        if time.time() - start_time > timeout:
            proc.terminate()

            error_msg = f"vLLM server startup timed out after {timeout}s"

            result = BenchResult(
                config=cfg,
                returncode=-1,
                runtime_sec=0,
                metrics={},
                error_msg=error_msg
            )

            task.result = result
            task.status = "failed"

            write_task_log(task_id, "[START SERVER]" + error_msg)

            raise RuntimeError(error_msg)

        time.sleep(1)



def serve_then_bench(task: BenchTask, port: int, host: str = "127.0.0.1"):
    """main execution workflow unit of the entire system,
    first starts the vlLM server by running the vllm serve command, 
    then benchmarks it by running the vllm bench command 
    """

    cfg = task.config
    gpu_ids = task.gpu_assigned
    task_id = task.id

    server_proc = None
    gpu_samples = []
    stop_event = threading.Event()

    try:

        server_proc = start_vllm_server(task, gpu_ids, port, host)

        sampler_thread = threading.Thread(
            target=sample_gpu_stats,
            args=(task_id, gpu_ids, stop_event, gpu_samples),
            daemon=True
        )
        sampler_thread.start()

        cmd = [
            "vllm", "bench", "serve",
            "--backend", cfg.backend,
            "--model", cfg.model_name,
            "--host", host,
            "--port", str(port),
        ]

        #if more parameters are added; those can be simply added here

        if cfg.num_prompts:
            cmd += ["--num-prompts", str(cfg.num_prompts)]
        if cfg.input_len:
            cmd += ["--random-input-len", str(cfg.input_len)]
        if cfg.output_len:
            cmd += ["--random-output-len", str(cfg.output_len)]
        if cfg.max_concurrency:
            cmd += ["--max-concurrency", str(cfg.max_concurrency)]
        if cfg.request_rate:
            cmd+= ["--request-rate", str(cfg.request_rate)]


        #more percentile metrics or other config to be added is to be done here
        cmd += ["--percentile-metrics", "e2el,tpot,ttft,itl"]
        cmd += ["--metric-percentiles", "95,99"]

        write_task_log(task_id, f"[SERVE THEN BENCH] CMD: {' '.join(cmd)}")

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

        if proc.returncode != 0:
            error_msg = f"Benchmark failed:\n{stderr[-1000:]}"

            result = BenchResult(
                config=cfg,
                returncode=proc.returncode,
                runtime_sec=duration,
                metrics={},
                error_msg=error_msg
            )

            task.result = result
            task.status = "failed"

            write_task_log(task_id, "[SERVE THEN BENCH]" + error_msg)

            raise RuntimeError(error_msg)

        metrics = parse_metrics(task_id, stdout)
        if metrics.get("successful_requests", 0) == 0:
            error_msg = "Benchmark ran but no successful requests"

            result = BenchResult(
                config=cfg,
                returncode=-1,
                runtime_sec=duration,
                metrics=metrics,
                error_msg=error_msg
            )

            task.result = result
            task.status = "failed"

            write_task_log(task_id, "[SERVE THEN BENCH] " + error_msg)

            return

        # GPU metrics aggregation
        if gpu_samples:
            per_gpu_metrics = {}
            all_utils = []
            all_mems = []

            for gid in gpu_ids:
                util_vals = []
                mem_vals = []

                for sample in gpu_samples:
                    gpu_data = sample["gpus"].get(str(gid))
                    if gpu_data:
                        util_vals.append(gpu_data["gpu_util_percent"])
                        mem_vals.append(gpu_data["mem_used_mb"])

                if util_vals:
                    per_gpu_metrics[str(gid)] = {
                        "avg_gpu_util_percent": sum(util_vals) / len(util_vals),
                        "peak_gpu_util_percent": max(util_vals),
                        "avg_gpu_mem_mb": sum(mem_vals) / len(mem_vals),
                        "peak_gpu_mem_mb": max(mem_vals),
                    }

                    all_utils.extend(util_vals)
                    all_mems.extend(mem_vals)

            metrics["gpu_metrics"] = per_gpu_metrics

        bench_logs = write_benchmark_log(task, duration, proc.returncode, metrics, stderr, stdout)
        write_task_log(task_id, "[SERVE THEN BENCH] Bench Successful")

        result = BenchResult(
            config=cfg,
            returncode=proc.returncode,
            runtime_sec=duration,
            metrics=metrics,
            error_msg=""
        )
        #mutating the task to store proper result, bench_logs and status; 
        #so that UI can render these

        task.result = result
        task.result.bench_logs = bench_logs
        process_logs = get_last_log_block("process.log")
        task.result.process_logs = process_logs
        task.status = "completed" #update status at the last so that UI doesnt misinterpret
    
        return

    except Exception as e:
        if not task.result:
            error_msg = str(e)

            result = BenchResult(
                config=cfg,
                returncode=-1,
                runtime_sec=0,
                metrics={},
                error_msg=error_msg
            )

            task.result = result

        task.status = "failed"

        write_task_log(task_id, f"[SERVE THEN BENCH] FATAL ERROR {str(e)}")

        return   

    finally:
        if server_proc:
            write_task_log(task_id, "[CLEANUP] Terminating server")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()