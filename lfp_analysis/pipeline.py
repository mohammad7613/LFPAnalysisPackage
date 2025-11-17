import numpy as np
from typing import Dict, List, Any
from lfp_analysis.registry.base import REGISTRIES
import warnings
# Turn numpy warnings into exceptions so debugger can catch them
np.seterr(all='raise')
warnings.filterwarnings('error')

class LfpPipeline:
    """
    A dependency-graph pipeline for LFP analysis.

    Executes datasets → preprocessors → features → visualizers
    according to a declarative config.
    """

    def __init__(self, datasets_cfg, preprocessors_cfg, features_cfg, visualizers_cfg, storages_cfg):
        self.datasets_cfg = datasets_cfg
        self.preprocessors_cfg = preprocessors_cfg
        self.features_cfg = features_cfg
        self.visualizers_cfg = visualizers_cfg
        self.storages_cfg = storages_cfg
        # Internal containers
        self.datasets: Dict[str, Any] = {}
        self.preprocessors: Dict[str, Any] = {}
        self.features: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BUILD STAGE
    # ------------------------------------------------------------------
    def build(self):
        """Instantiate all objects based on the config."""
        # 1. Load datasets
        for d in self.datasets_cfg:
            loader_cls = REGISTRIES["loaders"][d["format"]]
            self.datasets[d["id"]] = loader_cls()(d["path"])

        # 2. Instantiate preprocessors
        for p in self.preprocessors_cfg:
            cls = REGISTRIES["preprocessors"][p["name"]]
            self.preprocessors[p["id"]] = cls(**p.get("args", {}))

        # 3. Prepare features (not executed yet)
        self.features = {
            f["id"]: {
                "spec": f,
                "instance": REGISTRIES["features"][f["name"]](**f.get("args", {})),
                "result": None,
            }
            for f in self.features_cfg
        }

    # ------------------------------------------------------------------
    # EXECUTION STAGE
    # ------------------------------------------------------------------
    def run(self):
        """Execute the pipeline graph."""
        # --- Compute features ---
        for fid, fdict in self.features.items():
            spec = fdict["spec"]
            dataset_id = spec["dataset"]

            # get raw data from dataset
            data = self.datasets[dataset_id].copy()
            
            if isinstance(data, dict):
                # optional preprocessors
                signal = data["signal"]
                for pid in spec.get("preprocessors", []):
                    preproc = self.preprocessors[pid]
                    signal = preproc.process(signal)
                data["signal"] = signal

                #compute feature
                fdict["result"] = fdict["instance"].compute(**data)
            else:
                # optional preprocessors
                for pid in spec.get("preprocessors", []):
                    preproc = self.preprocessors[pid]
                    data = preproc.process(data)
                
                #compute feature
                fdict["result"] = fdict["instance"].compute(signal=data)
                
            self.results[fid] = fdict["result"]

        # --- Run visualizers ---
        for vis_spec in self.visualizers_cfg:
            vis_cls = REGISTRIES["visualizers"][vis_spec["name"]]
            vis = vis_cls(vis_spec.get("args", {}))

            # gather all input feature results
            input_features = [
                {"id": inp["feature"], "data": self.results[inp["feature"]]}
                for inp in vis_spec.get("inputs", [])
            ]

            vis.visualize(input_features)
        
        for saver_spec in self.storages_cfg:
            saver_cls = REGISTRIES["storages"][saver_spec["name"]]
            saver = saver_cls(saver_spec.get("args", {}))

            # gather all input feature results
            input_features = [
                {"id": inp, "data": self.results[inp]}
                for inp in saver_spec.get("args", [])
            ]
            saver.store(input_features)


    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------
    def summary(self):
        """Print summary of the pipeline graph."""
        print("=== Pipeline Summary ===")
        print(f"Datasets: {list(self.datasets.keys())}")
        print(f"Preprocessors: {list(self.preprocessors.keys())}")
        print(f"Features: {list(self.features.keys())}")
        print("Visualizers:")
        for v in self.visualizers_cfg:
            inputs = [i["feature"] for i in v.get("inputs", [])]
            print(f"  - {v['name']}")
            for input in inputs:
                 print(f"  - ... <- {input} <- {self.features[input]["spec"]["preprocessors"]} <- {self.features[input]["spec"]["dataset"]}")
        print("=========================")

