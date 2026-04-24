from pydantic import BaseModel
from typing import Dict


class Config(BaseModel):
    history_file: str
    access_code: str
    router_port: int
    gpu_backends: Dict[str, str]
    n_available_gpus: Dict[str, int]

