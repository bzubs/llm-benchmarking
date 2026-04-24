from pydantic import BaseModel
from typing import Optional, Literal, List, Dict, Any
from enum import Enum
from datetime import datetime


class DType(str, Enum):
    auto = "auto"
    fp16 = "fp16"
    fp32 = "fp32"
    bfloat16 = "bfloat16"


class GPUType(str, Enum):
    H100 = "H100"
    L40 = "L40"
    L4 = "L4"


class Quant(str, Enum):
    int8 = "int8"
    int4 = "int4"
    fp8 = "fp8"
    gptq = "gptq"
    awq = "awq"
    none = "none"


class BenchmarkConfig(BaseModel):

    # mode and type both deprecated in v2+
    benchmark_mode: str = "Online (Real-time Serving)"
    benchmark_type: Literal["serve"] = "serve"

    # model
    model_name: str
    dtype: DType
    max_model_len: Optional[int] = 256
    quantization: Quant

    # prompt
    dataset_name: Literal["random", "sharegpt", "custom"] = "random"
    input_len: Optional[int] = 128
    output_len: Optional[int] = 128

    # workload
    num_prompts: Optional[int] = 5
    max_concurrency: Optional[int] = 2
    request_rate: Optional[int] = None

    # backend
    backend: Optional[str] = "vllm"
    endpoint: Optional[str] = "/v1/completions"
    host: str = "localhost"
    port: str = "8000"

    # gpu
    gpu_type: GPUType
    gpu_memory_util: Optional[float] = 0.85
    n_gpus_required: int = 1
    dp_size: int = 1
    tp_size: int = 1


class GPU(BaseModel):
    id: int
    status: Literal["busy", "free"]


class BenchResult(BaseModel):
    returncode: int
    runtime_sec: float
    metrics: Optional[dict] = None
    error_msg: Optional[str] = ""
    process_logs: Optional[str] = ""
    bench_logs: Optional[str] = ""


class BenchTask(BaseModel):
    id: str = ""
    username: str
    config: BenchmarkConfig

    status: Literal["init", "assigned", "running", "completed", "queued", "failed"] = (
        "init"
    )

    gpu_assigned: Optional[List[int]] = None
    result: Optional[BenchResult] = None

    # timestamps
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()


class GPUNode(BaseModel):
    gpu: GPU


class BenchTaskResponse(BaseModel):
    id: int
    gpu_assigned: Optional[List[int]]
    status: str
    result: Optional[BenchResult] = None
