
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from .preprocess import Preprocessor
from .feature import FeatureFunction
from .visualization import Visualizer

class LfpPipeline:
    def __init__(self, preprocessors: List[Preprocessor], 
                 feature_functions: List[FeatureFunction], 
                 visualizers: List[Visualizer]):
        self.preprocessors = preprocessors
        self.feature_functions = feature_functions
        self.visualizers = visualizers

    def run(self, data: np.ndarray):
        # Step 1: Preprocessing
        for preproc in self.preprocessors:
            data = preproc.process(data)

        # Step 2: Feature extraction
        features = []
        for function in self.feature_functions:
            feat = {}
            feat["features"] = function.compute(data)
            features.append(feat)

        # Step 3: Visualization
        for k,vis in enumerate(self.visualizers):
            vis.run(features[k])
