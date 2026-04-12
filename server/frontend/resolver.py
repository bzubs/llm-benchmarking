from dataclasses import dataclass
from typing import List
from transformers import AutoConfig
from huggingface_hub import list_repo_files


# Data Model
@dataclass
class ModelCapabilities:
    max_model_len: int
    supported_dtypes: List[str]
    supported_quantizations: List[str]
    default_dtype: str
    default_quantization: str


# Capability Resolver
class CapabilityResolver:

    # Public Entry Point
    def resolve(self, model_name: str) -> ModelCapabilities:
        max_len = self._resolve_max_len(model_name)

        # Detect quantization first (checkpoint-level)
        supported_quant = self._resolve_quantization(model_name)

        # Resolve dtype based on GPU + quantization constraints
        supported_dtypes = self._resolve_dtypes(supported_quant)

        default_dtype = "auto"

        default_quant = supported_quant[0]

        return ModelCapabilities(
            max_model_len=max_len,
            supported_dtypes=supported_dtypes,
            supported_quantizations=supported_quant,
            default_dtype=default_dtype,
            default_quantization=default_quant,
        )

    # MODEL METADATA (Context Length)
    def _resolve_max_len(self, model_name: str) -> int:
        try:
            config = AutoConfig.from_pretrained(model_name)
            return getattr(config, "max_position_embeddings", 8192)
        except Exception:
            return 8192

    # CHECKPOINT → QUANTIZATION LOGIC

    def _resolve_quantization(self, model_name: str) -> List[str]:
        quant = []
        name_lower = model_name.lower()

        try:
            files = list_repo_files(model_name)
        except Exception:
            return ["none"]

        # --- Repo name detection ---
        if "gptq" in name_lower:
            quant.append("gptq")

        if "awq" in name_lower:
            quant.append("awq")

        # --- File-based detection ---
        if "quantize_config.json" in files:
            quant.append("gptq")

        if any("awq" in f.lower() for f in files):
            quant.append("awq")

        if not quant:
            quant.append("none")

        return list(set(quant))

    # GPU → DTYPE LOGIC
    def _resolve_dtypes(self, quantizations: List[str]) -> List[str]:

        # If checkpoint is pre-quantized, dtype must be auto
        if any(q in ["gptq", "awq"] for q in quantizations):
            return ["auto"]

        supported = ["auto", "float16", "fp8", "bfloat16"]

        # Hopper
        # if major >= 9:
        #     supported += ["float16", "bfloat16", "fp8"]

        # # Ampere
        # elif major >= 8:
        #     supported += ["float16", "bfloat16"]

        # # Volta / Turing
        # else:
        #     supported += ["float16"]

        return supported
