import os
from schema import BenchmarkConfig


def build_cli(cfg: BenchmarkConfig) -> list:
    # For online (serve) benchmarks
    if cfg.benchmark_type == "serve":
        base = ["vllm", "bench", "serve"]
        
        base += ["--backend", cfg.backend]
        base += ["--model", cfg.model_name]
        base += ["--endpoint", cfg.endpoint]
        
        # Dataset configuration
        base += ["--dataset-name", cfg.dataset_name]
        
        if cfg.dataset_path:
            base += ["--dataset-path", cfg.dataset_path]
        
        if cfg.num_prompts:
            base += ["--num-prompts", str(cfg.num_prompts)]
        
        return base
    
    # For offline benchmarks
    base = ["vllm", "bench", cfg.benchmark_type]

    base += ["--model", cfg.model_name]
    if getattr(cfg, "dtype", None):
        base += ["--dtype", cfg.dtype]

    # synthetic prompt
    if cfg.dataset_name == "random" and cfg.benchmark_type=="latency":
        if getattr(cfg, "input_len", None):
            base += ["--input-len", str(cfg.input_len)]
        if getattr(cfg, "output_len", None):
            base += ["--output-len", str(cfg.output_len)]
    elif cfg.dataset_name =="random" and cfg.benchmark_type=="throughput":
        if getattr(cfg, "input_len", None):
            base += ["--random-input-len", str(cfg.input_len)]
        if getattr(cfg, "output_len", None):
            base += ["--random-output-len", str(cfg.output_len)]
    elif cfg.dataset_path:
        base += ["--dataset-path", cfg.dataset_path]

    if cfg.batch_size:
        base += ["--batch-size", str(cfg.batch_size)]

    if cfg.gen_sequences:
        base+=["--n", str(cfg.gen_sequences)]

    if cfg.benchmark_type=="latency" and cfg.num_iters:
        base+=["--num-iters", str(cfg.num_iters)]

    if cfg.benchmark_type=="throughput" and cfg.num_prompts:
        base +=["--num-prompts", str(cfg.num_prompts)]

    if cfg.max_model_len:
        base+=["--max-model-len", str(cfg.max_model_len)]

    if cfg.quantization:
        base+=["--quantization", str(cfg.quantization)]

    base += ["--gpu-memory-utilization", str(cfg.gpu_memory_util)]
    base += ["--output-json", str(cfg.output_json)]

    return base
 
def env_for_gpu(gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env

