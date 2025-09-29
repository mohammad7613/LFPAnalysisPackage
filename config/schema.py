# lfp_analysis/config/schema.py

def validate_config(config: dict):
    """Minimal validation of YAML structure."""
    if "dataset" not in config:
        raise ValueError("Config must define a 'dataset' section.")
    if "path" not in config["dataset"]:
        raise ValueError("Dataset section must include 'path'.")

    # Optional checks
    for section in ["preprocessors", "features", "visualizers"]:
        if section in config and not isinstance(config[section], list):
            raise ValueError(f"Section '{section}' must be a list of items.")

    return True
