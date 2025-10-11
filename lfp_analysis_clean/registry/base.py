# registry/base.py

REGISTRIES = {
    "preprocessors": {},
    "features": {},
    "visualizers": {},
    "loaders": {},   # optional: dataset loaders
    "statistics": {}
}

def register(registry_name: str, key: str):
    """
    Decorator to register a class/function under a registry.

    Example:
        @register("features", "band_power")
        class BandPower(FeatureFunction): ...
    """
    def decorator(cls):
        if registry_name not in REGISTRIES:
            raise ValueError(f"Unknown registry {registry_name}")
        REGISTRIES[registry_name][key] = cls
        return cls
    return decorator


def get_registry(registry_name: str):
    """Return the dictionary for a given registry."""
    return REGISTRIES[registry_name]
