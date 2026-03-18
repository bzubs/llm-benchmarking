from pydantic import BaseModel, Field
from typing import Optional, Literal, List


class BenchmarkConfig(BaseModel):
    # mode and type both deprecated in v2
    benchmark_mode: str = "Online (Real-time Serving)"
    benchmark_type: Literal["serve"] = "serve"
    
    #user
    username : str = None
    
    model_name: str
    dtype: Literal["auto", "fp16", "fp32", "bfloat16"] = "fp16"
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

    host : str = "localhost"
    port : str = "8023"


class GPU(BaseModel):
    id: int
    status : Literal["busy", "free"]



class BenchTask(BaseModel):
    id : int
    config : BenchmarkConfig
    status : Literal["init", "assigned", "running", "completed", "queued", "failed"]= "init"
    gpu_assigned : Optional[int] = None



class GPUNode(BaseModel):
    gpu: GPU
    queue: List[BenchTask] = Field(default_factory=list)
