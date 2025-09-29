from lfp_analysis.config import load_config
from lfp_analysis.registery import REGISTRIES
from lfp_analysis.pipeline import LfpPipeline

def build_from_yaml(config_path: str):
    config = load_config(config_path)

    # Load dataset
    loader_cls = REGISTRIES["loaders"][config["dataset"]["format"]]
    lfp = loader_cls()(config["dataset"]["path"])

    preprocessors = [REGISTRIES["preprocessors"][p["name"]](**p.get("args", {}))
                     for p in config.get("preprocessors", [])]
    features = [REGISTRIES["features"][f["name"]](**f.get("args", {}))
                for f in config.get("features", [])]
    visualizers = [REGISTRIES["visualizers"][v["name"]](**v.get("args", {}))
                   for v in config.get("visualizers", [])]

    return LfpPipeline(preprocessors, features, visualizers), lfp
