from lfp_analysis.config import load_config
from lfp_analysis.registry import REGISTRIES
from lfp_analysis.pipeline import LfpPipeline


def build_from_yaml(config_path: str):
    config = load_config(config_path)
    pipeline = LfpPipeline(
        config.get("datasets", []),
        config.get("preprocessors", []),
        config.get("features", []),
        config.get("visualizers", []),
        config.get("storages", [])
    )
    pipeline.build()
    return pipeline
