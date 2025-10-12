# lfp_analysis/config/loader.py
import yaml
from pathlib import Path
from .schema import validate_config

def load_config(path: str) -> dict:
    """Load YAML config into a Python dict and validate it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # optional: validate against schema
    # validate_config(config)

    return config
