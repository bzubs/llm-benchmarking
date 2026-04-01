from pydantic import BaseModel
from typing import List

"""defines Config Schema of the model of config.yaml"""

class Config(BaseModel):
    backend_port: int
    gpu_ids: List[int]
