import os
from schema import BenchmarkConfig


def build_cli(cfg: BenchmarkConfig) -> list:
    cmd = [
        "vllm", "bench", "serve",
        "--backend", cfg.backend,
        "--model", cfg.model_name,
        "--endpoint", cfg.endpoint,
        "--host", cfg.host,
        "--port", cfg.port,
    ]

    if cfg.num_prompts:
        cmd += ["--num-prompts", str(cfg.num_prompts)]
    if cfg.input_len:
        cmd += ["--random-input-len", str(cfg.input_len)]
    if cfg.output_len:
        cmd += ["--random-output-len", str(cfg.output_len)]
    if cfg.max_concurrency:
        cmd += ["--max-concurrency", str(cfg.max_concurrency)]

    return cmd
    
 
def env_for_gpu(gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env

