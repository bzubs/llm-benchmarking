# v1 Model configurations
MODEL_CONFIGS = {
    "gpt2": {
        "max_context": 1024,
        "params": "125M",
        "description": "Small GPT-2 model for testing"
    },
    "NousResearch/Hermes-3-Llama-3.1-8B": {
        "max_context": 131072,
        "params": "8B",
        "description": "Hermes 3 Llama 3.1 8B model"
    }
}

#not integrated 
PRESET_CONFIGS = {
    "Quick Test": {
        "num_concurrency": 1,
        "num_prompts": 5,
        "input_len": 32,
        "output_len": 32,
    },
    "Standard": {
        "num_concurrency": 10,
        "num_prompts": 5,
        "input_len": 128,
        "output_len": 128,
    },
    "Heavy Load": {
        "num_prompts": 20,
        "num_concurrency": 50,
        "input_len": 1024,
        "output_len": 1024,
    }
}
