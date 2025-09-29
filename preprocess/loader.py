# lfp_analysis/data_access/loader.py
import numpy as np
import scipy.io as sio
from lfp_analysis.registery import register

class BaseLoader:
    def __call__(self, path: str):
        raise NotImplementedError

@register("loaders", "npy")
class NpyLoader(BaseLoader):
    """Load LFP dataset stored as .npy"""
    def __call__(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        # expected keys: lfp, stim, time, sfreq
        return data["lfp"], data["stim"], data["time"], data["sfreq"]

@register("loaders", "mat")
class MatLoader(BaseLoader):
    """Load LFP dataset stored as .mat"""
    def __call__(self, path: str):
        EpochedData = sio.loadmat(path)
        LFPdata = EpochedData["epoched_data_config"]["data"][0][0] 
        return LFPdata
