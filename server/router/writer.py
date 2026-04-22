from datetime import datetime
import json
from schemas.task import BenchTask


def append_jsonl_history(path: str, task: BenchTask):
    if task and task.result:
        metrics = task.result.metrics
        if metrics:

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "taskID": task.id,
                "task_status": task.status,
                "username": task.username,
                "model": task.config.model_name,
                "gpu": task.gpu_assigned,
                "runtime_sec": task.result.runtime_sec,
                "returncode": task.result.returncode,
                # model config
                "max_model_len": task.config.max_model_len,
                "quantization": task.config.quantization,
                "dtype": task.config.dtype,
                # prompt and workload config
                "num_prompts": task.config.num_prompts,
                "max_concurrency": task.config.max_concurrency,
                "input_len": task.config.input_len,
                "output_len": task.config.output_len,
                "request_rate": task.config.request_rate,
                # host, port
                "host": task.config.host,
                "port": task.config.port,
                # gpu config
                "gpu_type": task.config.gpu_type,
                "gpu_memory_util": task.config.gpu_memory_util,
                "n_gpus_required": task.config.n_gpus_required,
                "dp_size": task.config.dp_size,
                "tp_size": task.config.tp_size,
                # Core metrics (safe access)
                "successful_requests": metrics.get("successful_requests"),
                "request_throughput": metrics.get("request_throughput"),
                "total_token_throughput": metrics.get("total_token_throughput"),
                "median_ttft_ms": metrics.get("median_ttft_ms"),
                "median_tpot_ms": metrics.get("median_tpot_ms"),
                "median_itl_ms": metrics.get("median_itl_ms"),
                "median_e2el_ms": metrics.get("median_e2el_ms"),
                #     # Useful config snapshot
                #     "avg_gpu_util_percent": metrics.get("avg_gpu_util_percent"),
                #     "peak_gpu_util_percent": metrics.get("peak_gpu_util_percent"),
                #     "avg_gpu_mem_mb": metrics.get("avg_gpu_mem_mb"),
                #     "peak_gpu_mem_mb": metrics.get("peak_gpu_mem_mb"),
            }

        else:
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "taskID": task.id,
                "task_status": task.status,
                "username": task.username,
                "model": task.config.model_name,
                "gpu": task.gpu_assigned,
                "runtime_sec": task.result.runtime_sec,
                "returncode": task.result.returncode,
                # Useful config snapshot
                "num_prompts": task.config.num_prompts,
                "max_concurrency": task.config.max_concurrency,
                "input_len": task.config.input_len,
                "output_len": task.config.output_len,
                "quantization": task.config.quantization,
                "dtype": task.config.dtype,
                "gpu_memory_util": task.config.gpu_memory_util,
            }

        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
