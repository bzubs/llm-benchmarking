from pydantic import BaseModel
from typing import Optional, Literal


class BenchmarkConfig(BaseModel):
    # mode
    benchmark_mode: str = "Online (Real-time Serving)"
    benchmark_type: Literal["latency", "throughput", "serve"] = "serve"
    # device
    device: Literal["gpu7", "gpu6"] = "gpu7"
    # model
    model_name: str
    dtype: Optional[Literal["auto", "float16", "float32", "bfloat16"]] = "fp16"
    max_model_len : Optional[int] = 256
    quantization: Literal["int8", "int4", "fp8", "gptq", "awq", "none"] = "none"
    # prompt
    dataset_name: Literal["random", "sharegpt", "custom"] = "random"
    input_len: Optional[int] = 128
    output_len: Optional[int] = 128
    dataset_path: Optional[str] = None
    # workload
    num_prompts: Optional[int] = 5
    max_concurrency: Optional[int] = 2

    backend: Optional[str] = "vllm"
    endpoint: Optional[str] = "/v1/completions"

    #gpu_memory_util
    gpu_memory_util : Optional[float] = 0.85
