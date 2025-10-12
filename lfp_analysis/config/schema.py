# lfp_analysis/config/schema.py

def validate_config(config: dict):
    """Minimal validation of YAML structure."""
    if "datasets" not in config:
        raise ValueError("Config must define a 'datasets' section.")
    if "path" not in config["datasets"]:
        raise ValueError("Dataset section must include 'path'.")

    # Optional checks
    for section in ["preprocessors", "features", "visualizers"]:
        if section in config and not isinstance(config[section], list):
            raise ValueError(f"Section '{section}' must be a list of items.")

    return True
