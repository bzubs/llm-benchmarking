import yaml
from config_schema import Config

"""simply parses the default config.yaml and returns Config Pydantic Model"""

def load_config(path="config.yaml") -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)

