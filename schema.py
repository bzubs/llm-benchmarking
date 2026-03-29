from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any


class BenchmarkConfig(BaseModel):

    # mode and type both deprecated in v2+
    benchmark_mode: str = "Online (Real-time Serving)"
    benchmark_type: Literal["serve"] = "serve"
    
    #user
    username : str = "none"
    
    #model
    model_name: str
    dtype: Literal["auto", "fp16", "fp32", "bfloat16"] = "fp16"
    max_model_len : Optional[int] = 256
    quantization: Literal["int8", "int4", "fp8", "gptq", "awq", "none"] = "none"
    
    # prompt
    dataset_name: Literal["random", "sharegpt", "custom"] = "random"
    input_len: Optional[int] = 128
    output_len: Optional[int] = 128
    

    # workload
    num_prompts: Optional[int] = 5
    max_concurrency: Optional[int] = 2
    request_rate : Optional[int] = None

    #backend
    backend: Optional[str] = "vllm"
    endpoint: Optional[str] = "/v1/completions"
    host : str = "localhost"
    port : str = "8000"

    #gpu
    gpu_type : Literal["H100", "L40", "L4"] = "H100"
    gpu_memory_util : Optional[float] = 0.85
    n_gpus_required : int = 1
    dp_size : int = 1
    tp_size : int = 1

    

class GPU(BaseModel):
    id: int
    status : Literal["busy", "free"]

class BenchResult(BaseModel):
    config: BenchmarkConfig
    returncode: int
    runtime_sec: float
    metrics: Optional[dict] = None
    error_msg : Optional[str] = ""


class BenchTask(BaseModel):
    id : int
    config : BenchmarkConfig
    status : Literal["init", "assigned", "running", "completed", "queued", "failed"]= "init"
    gpu_assigned : Optional[List[int]] = None
    result : Optional[BenchResult] = None



class GPUNode(BaseModel):
    gpu: GPU

class BenchTaskResponse(BaseModel):
    id: int
    gpu_assigned: Optional[List[int]]
    status: str
    result : Optional[BenchResult] = None