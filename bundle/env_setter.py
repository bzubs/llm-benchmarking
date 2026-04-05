import os
from schema import BenchmarkConfig

"""
env_for_gpu() : Sets environment for the process with the required GPU, 
build_cli() : Builds command for execution (deprecated in v2+)
"""

def env_for_gpu(gpu_ids) -> dict:
    env = os.environ.copy()

    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]

    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    return env



#dperecated in v2+; functions now build their own custom commands
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
    
 
