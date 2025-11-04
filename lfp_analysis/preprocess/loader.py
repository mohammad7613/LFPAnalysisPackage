# lfp_analysis/data_access/loader.py
import numpy as np
import scipy.io as sio
from lfp_analysis.registry import register

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


@register("loaders", "mat_cell")
class MatLoader(BaseLoader):
    """Load LFP dataset stored as .mat"""
    def __call__(self, path: str):
        ComplexityRestData = sio.loadmat(path)
        ComplexityRestDataArray = ComplexityRestData["complexityDataStandardChs"]["data"][0][0] 
        return ComplexityRestDataArray



@register("loaders", "mat_events")
class MatLoader(BaseLoader):
    """Load LFP dataset stored as .mat"""
    def __call__(self, path: str):
        EpochedData = sio.loadmat(path)
        LFPdata = EpochedData["epoched_data_config"]["data"][0][0]
        events = self.extract_event_matrix(EpochedData=EpochedData)
        data = {'signal': LFPdata, 'events': events}
        return data
    
    def extract_event_matrix(self,EpochedData, n_epochs=120):
        """
        Extracts trial type matrix of shape (n_sessions, n_epochs)
        from epoched stimuli data.
        """

        stimuli_data = EpochedData["epoched_data_config"]['stimuli_data'][0][0]

        n_sessions = len(stimuli_data)
        event_matrix = np.zeros((n_sessions, n_epochs), dtype=np.int32)

        for i in range(n_sessions):
            trial_types = stimuli_data[i][0].squeeze()   # shape: (2*n_epochs_total,)
            event_matrix[i] = trial_types[0,:n_epochs]     # select first 100 trials

        return event_matrix