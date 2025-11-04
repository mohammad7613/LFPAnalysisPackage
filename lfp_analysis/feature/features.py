# features/power_band.py
# lfp_analysis/business_logic/spectral.py
"""Spectral feature implementations (example: band power)."""
from typing import Tuple
import numpy as np
from .base import FeatureFunction
from scipy.signal import welch
from npeet import entropy_estimators as ee
from typing import List, Tuple
from lfp_analysis.registry import register
from scipy.signal import butter, filtfilt, hilbert,firwin, correlate
from lfp_analysis.registry.base import REGISTRIES
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed


#### Features Utils
# These are features that has been applied to single singal 
@register("features","Complexity_base")
class ComplexityBase(FeatureFunction):
    def __init__(self,complexity_type: str, arguments: dict):
        self.complexity_type = complexity_type
        self.complexity_types = {"SampEn": self.SampEn, "FuzzyEn": self.FuzEn, "LZC": self.LZC}
        self.argument = arguments
        assert complexity_type in self.complexity_types.keys(), f"Complexity type {complexity_type} not recognized. Available types: {list(self.complexity_types.keys())}"
        self.complexity_types_args = {"SampEn": ["m", "r", "tau"], "FuzzyEn": ["m", "r", "n", "tau"], "LZC": ["num_levels"]}
        assert set(self.argument.keys()) == set(self.complexity_types_args[complexity_type]), f"Arguments for {complexity_type} must be {self.complexity_types_args[complexity_type]}"   
    def compute(self, signal, **kwargs):
        return self.complexity_types[self.complexity_type](x=signal,argument=self.argument)

    def SampEn(self,x,argument):
        """
        Sample Entropy (SampEn) of a univariate signal x.

        Parameters
        ----------
        x : 1D array
            Input signal
        m : int
            Embedding dimension
        r : float
            Threshold (commonly 0.15 * std(x))
        tau : int, optional
            Time lag (default = 1)

        Returns
        -------
        Out_SampEn : float
            The Sample Entropy of x
        P : list
            [total template matches of length m, total forward matches of length m+1]
        """
        m = argument["m"]
        tau = argument["tau"]
        r = argument["r"]

        if tau > 1:
            x = x[::tau]  # downsample

        N = len(x)
        P = np.zeros(2)
        xMat = np.zeros((m+1, N-m))

        for i in range(m+1):
            xMat[i, :] = x[i:N-m+i]

        for k in range(m, m+2):
            count = np.zeros(N-m)
            tempMat = xMat[:k, :]

            for i in range(N-k):
                # Chebyshev distance (max abs difference)
                dist = np.max(np.abs(tempMat[:, i+1:N-m] - tempMat[:, [i]]), axis=0)
                D = (dist < r).astype(int)
                count[i] = np.sum(D) / (N-m)

            P[k-m] = np.sum(count) / (N-m)

        Out_SampEn = np.log(P[0] / P[1])
        return Out_SampEn

    def FuzEn(self, x, argument):
        """
        Fuzzy Entropy (FuzEn) of a univariate signal x.

        Parameters
        ----------
        x : 1D array-like
            Input signal
        m : int
            Embedding dimension
        r : float
            Threshold (commonly 0.15 * std(x))
        n : float, optional
            Fuzzy power (default=2)
        tau : int, optional
            Time lag (default=1)

        Returns
        -------
        Out_FuzEn : float
            The Fuzzy Entropy of x
        P : list
            [global quantity in dimension m, global quantity in dimension m+1]
        """
        m = argument["m"]
        tau = argument["tau"]
        r = argument["r"]
        n = argument["n"]   

        x = np.asarray(x, dtype=float)

        if tau > 1:
            x = x[::tau]  # downsample

        N = len(x)
        P = np.zeros(2)
        xMat = np.zeros((m+1, N-m))

        # Construct template matrix
        for i in range(m+1):
            xMat[i, :] = x[i:N-m+i]

        # Loop for m and m+1
        for k in range(m, m+2):
            count = np.zeros(N-m)
            tempMat = xMat[:k, :]

            for i in range(N-k):
                # Chebyshev distance (max abs difference)
                dist = np.max(np.abs(tempMat[:, i+1:N-m] - tempMat[:, [i]]), axis=0)
                # Fuzzy membership function
                DF = np.exp(-(dist**n) / r)
                count[i] = np.sum(DF) / (N-m-1)

            P[k-m] = np.sum(count) / (N-m)

        Out_FuzEn = np.log(P[0] / P[1])
        return Out_FuzEn
    def LZC(self,x, argument):
        """
        Lempel-Ziv complexity of a univariate signal.

        Parameters
        ----------
        signal : 1D array-like
            Input signal
        num_levels : int
            Quantization levels (2 or 3)

        Returns
        -------
        out : float
            Lempel-Ziv complexity
        P : ndarray
            Quantized symbolic sequence
        """
        num_levels = argument["num_levels"]
        signal = x
        signal = np.asarray(signal, dtype=float)
        med = np.median(signal)

        # Quantization step
        if num_levels == 2:
            P = ((np.sign(signal - med) + 1) / 2).astype(int)
        elif num_levels == 3:
            P = np.zeros_like(signal, dtype=int)
            max_val = np.abs(np.max(signal))
            P[signal >= med + max_val / 16] = 2
            P[signal <= med - max_val / 16] = 0
            mask = (signal > med - max_val / 16) & (signal < med + max_val / 16)
            P[mask] = 1
        else:
            raise ValueError("num_levels must be 2 or 3")

        c = 2
        terminate = False
        r = 1
        i = 1

        while not terminate:
            S = P[:r]
            Q = P[r:r+i]

            # Check if Q exists in S+Q (excluding last symbol)
            concat = np.concatenate([S, Q])
            found = False
            if len(Q) > 0:
                for j in range(len(concat) - len(Q)):
                    if np.array_equal(concat[j:j+len(Q)], Q):
                        found = True
                        break

            if not found:
                c += 1
                r = r + i
                i = 1
            else:
                i += 1

            if r + i == len(P):
                terminate = True

        out = c * np.log2(len(P)) / len(P)
        return out


@register("features","DPLI_base")
class DPLI(FeatureFunction):
    """

    """
    def __init__(self, sfreq: float, x_lowcut: float, x_highcut: float, y_lowcut: float, y_highcut:float):
        self.fs = sfreq
        self.lowcut_x = x_lowcut
        self.highcut_x = x_highcut
        self.lowcut_y = y_lowcut
        self.highcut_y = y_highcut

    def compute(self, signal1: np.ndarray, signal2: np.ndarray, **kwargs):
        return self.compute_dpli_band(signal_x=signal1,signal_y=signal2,fs=self.fs,lowcut_x = self.lowcut_x,highcut_x = self.highcut_x,lowcut_y=self.lowcut_y,highcut_y=self.highcut_y)
        
    def butter_bandpass(self,lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return b, a

    def bandpass_filter(self,data, lowcut, highcut, fs, order=4):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        return filtfilt(b, a, data)

    def compute_dpli_band(self,signal_x, signal_y, fs, lowcut_x, highcut_x,lowcut_y, highcut_y):
        # Step 1: filter each signal into precise band
        x_f = self.bandpass_filter(signal_x, lowcut_x, highcut_x, fs)
        y_f = self.bandpass_filter(signal_y, lowcut_y, highcut_y, fs)

        # Step 2: compute phases via Hilbert
        ph_x = np.angle(hilbert(x_f))
        ph_y = np.angle(hilbert(y_f))

        # Step 3: wrap phase differences
        delta = np.angle(np.exp(1j * (ph_x - ph_y)))

        # Step 4: compute dPLI
        return np.mean(delta > 0)


@register("features","TE_base")
class TE(FeatureFunction):
    def __init__(self,d_x: int, d_y: int, w_x: int, w_y: int):
        self.d_x = d_x
        self.d_y = d_y
        self.w_x = w_x
        self.w_y = w_y


    def compute(self, signal1: np.ndarray, signal2: np.ndarray, **kwargs):
        return self.TransferEntropy(X=signal1,Y=signal2,d_x=self.d_x,d_y=self.d_y,w_x=self.w_x,w_y=self.w_y)
        

    def generate_lagged_vectors(self, signal: np.ndarray, lag: int) -> np.ndarray :
        """Generate lagged vectors from a signal."""
        n = len(signal)
        lagged_vectors = np.array([signal[i: n - lag + i + 1] for i in range(lag)]).T
        return lagged_vectors


    def TransferEntropy(self, X: np.ndarray, Y: np.ndarray, d_x: int, d_y: int, w_x: int, w_y: int) -> float:
        """
        Calculate the Transfer Entropy from X to Y.
        
        Parameters:
        - Y (np.ndarray): The target signal.
        - X (np.ndarray): The source signal.
        - d_y (int): The lag for the target signal.
        - d_x (int): The lag for the source signal.
        - w_y (int): The window size for the target signal.
        - w_x (int): The window size for the source signal.

        Returns:
        - float: The calculated Transfer Entropy value.
        """

        X_lagged = self.generate_lagged_vectors(X, w_x)
        Y_lagged = self.generate_lagged_vectors(Y, w_y)
        X_lagged = X_lagged[:-d_x-1]
        Y_lagged = Y_lagged[:-d_y-1]

        # Remove the initial part where lagged vectors are not defined
        max_index = max(w_x + d_x , w_y + d_y)
        if w_x + d_x == max_index:
            Y_lagged = Y_lagged[max_index- w_y - d_y:]
        else:         
            X_lagged = X_lagged[max_index- w_x - d_x:]
        

        Y_t = Y[max_index:]
        
        return ee.cmi(Y_t, X_lagged, Y_lagged)






@register("features","TE")
class TE(FeatureFunction):
    def __init__(self,d_x: int, d_y: int, w_x: int, w_y: int, channel_pairs: List[Tuple], time_window: Tuple):
        self.d_x = d_x
        self.d_y = d_y
        self.w_x = w_x
        self.w_y = w_y
        self.channel_pairs = channel_pairs
        self.time_window = time_window


    def compute(self, signal, **kwargs) -> np.ndarray:
        TE = np.zeros((signal.shape[0],len(self.channel_pairs),signal.shape[2]))
        for k,ch_pair in enumerate(self.channel_pairs):
            ch1 = ch_pair[0]
            ch2 = ch_pair[1]
            for session in np.arange(signal.shape[0]):
                for trial in np.arange(signal.shape[2]):
                    X = signal[session,ch1,trial,self.time_window[0]:self.time_window[1]]
                    Y = signal[session,ch2,trial,self.time_window[0]:self.time_window[1]]
                    TE[session,k,trial] = self.TransferEntropy(X=X,Y=Y,d_x=self.d_x,d_y=self.d_y,w_x=self.w_x,w_y=self.w_y)
        return TE
        

    def generate_lagged_vectors(self, signal: np.ndarray, lag: int) -> np.ndarray :
        """Generate lagged vectors from a signal."""
        n = len(signal)
        lagged_vectors = np.array([signal[i: n - lag + i + 1] for i in range(lag)]).T
        return lagged_vectors


    def TransferEntropy(self, X: np.ndarray, Y: np.ndarray, d_x: int, d_y: int, w_x: int, w_y: int) -> float:
        """
        Calculate the Transfer Entropy from X to Y.
        
        Parameters:
        - Y (np.ndarray): The target signal.
        - X (np.ndarray): The source signal.
        - d_y (int): The lag for the target signal.
        - d_x (int): The lag for the source signal.
        - w_y (int): The window size for the target signal.
        - w_x (int): The window size for the source signal.

        Returns:
        - float: The calculated Transfer Entropy value.
        """

        X_lagged = self.generate_lagged_vectors(X, w_x)
        Y_lagged = self.generate_lagged_vectors(Y, w_y)
        X_lagged = X_lagged[:-d_x-1]
        Y_lagged = Y_lagged[:-d_y-1]

        # Remove the initial part where lagged vectors are not defined
        max_index = max(w_x + d_x , w_y + d_y)
        if w_x + d_x == max_index:
            Y_lagged = Y_lagged[max_index- w_y - d_y:]
        else:         
            X_lagged = X_lagged[max_index- w_x - d_x:]
        

        Y_t = Y[max_index:]
        
        return ee.cmi(Y_t, X_lagged, Y_lagged)





@register("features","band_power")
class BandPowerFeature(FeatureFunction):
    """
    Band power computed with Welch's method.
    Parameters
    ----------
    band : Tuple[float, float]
    (low_hz, high_hz)
    sfreq : float
    sampling frequency
    """
    def __init__(self, band: Tuple[float, float], sfreq: float):
        self.band = band
        self.sfreq = sfreq
    def _bandpower_1d(self, signal: np.ndarray) -> float:
        freqs, psd = welch(signal, fs=self.sfreq, nperseg=min(256, len(signal)))
        mask = (freqs >= self.band[0]) & (freqs <= self.band[1])
         # Integrate PSD over band
        return float(np.trapezoid(psd[mask], freqs[mask]))
    def compute(self, data, **kwargs) -> np.ndarray:
        """Compute bandpower.
        If `signal` has shape (n_samples,) -> return float.
        If `signal` has shape (n_sessions, n_channels, n_epochs, n_samples) ->
        return (n_sessions, n_epochs) averaged across channels.
        """
        signal = data
        if signal.ndim == 1:
            return self._bandpower_1d(signal)
    # assume full dataset
        sessions, channels, epochs, samples = signal.shape
        out = np.zeros((sessions, epochs), dtype=float)
        for s in range(sessions):
            for e in range(epochs):
                # average channels for a single trial
                avg_signal = signal[s, :, e, :].mean(axis=0)
                out[s, e] = self._bandpower_1d(avg_signal)
        return out


@register("features","MUA")
class MUAFeature(FeatureFunction):
    """
    Multi-unit activity (MUA) feature.
    Parameters
    ----------
    sfreq : float
        Sampling frequency.
    low_hz : float
        Low cut-off frequency for bandpass filter.
    high_hz : float
        High cut-off frequency for bandpass filter.
    """
    def __init__(self, sfreq: float, low_hz: float, high_hz: float, window_ms: Tuple=(0,50)):
        self.sfreq = sfreq
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.window_ms = window_ms

    def compute(self, epoch_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        epoch_data: array (n_trials, n_samples) aligned to stimulus onset
        fs: sampling frequency
        window_ms: response window in ms
        """
        responses = []
        for trial in epoch_data:
            # 1. Bandpass filter
            filtered = self.bandpass_filter(trial, self.sfreq,self.low_hz,self.high_hz)
            
            # 2. Spike detection
            spikes = self.detect_spikes(filtered)
            
            # 3. Bin spikes
            binned = self.bin_spikes(spikes, len(trial), self.sfreq)
            
            # 4. Count spikes in 0–50 ms window
            start_bin = int(self.window_ms[0] * self.sfreq / 1000)
            end_bin   = int(self.window_ms[1] * self.sfreq / 1000)
            response = np.sum(binned[start_bin:end_bin])
            responses.append(response)
        
        return np.array(responses)



    def bandpass_filter(data, fs, lowcut=200, highcut=6000, order=3):
        """Butterworth bandpass filter."""
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, data)

    def detect_spikes(filtered_signal, threshold_factor=7):
        """Detect spikes using threshold based on MAD."""
        mad = np.median(np.abs(filtered_signal - np.median(filtered_signal)))
        threshold = threshold_factor * mad
        spike_indices = np.where(np.abs(filtered_signal) > threshold)[0]
        return spike_indices

    def bin_spikes(spike_indices, n_samples, fs, bin_size_ms=1):
        """Convert spike indices to binned spike train at 1 ms resolution."""
        bin_size = int(fs * bin_size_ms / 1000)  # samples per bin
        n_bins = int(np.ceil(n_samples / bin_size))
        spike_train = np.zeros(n_bins, dtype=int)
        bin_indices = spike_indices // bin_size
        bin_indices = bin_indices[bin_indices < n_bins]
        np.add.at(spike_train, bin_indices, 1)
        return spike_train



@register("features","DPLI")
class DPLI(FeatureFunction):
    """
    Multi-unit activity (MUA) feature.
    Parameters
    ----------
    sfreq : float
        Sampling frequency.
    low_hz : float
        Low cut-off frequency for bandpass filter.
    high_hz : float
        High cut-off frequency for bandpass filter.
    """
    def __init__(self, sfreq: float, x_lowcut: float, x_highcut: float, y_lowcut: float, y_highcut:float, channel_pairs: List[Tuple],time_window: Tuple=(0,50)):
        self.fs = sfreq
        self.lowcut_x = x_lowcut
        self.highcut_x = x_highcut
        self.lowcut_y = y_lowcut
        self.highcut_y = y_highcut
        self.channel_pairs = channel_pairs
        self.time_window = time_window

    def compute(self, signal, **kwargs):
        DPLI = np.zeros((signal.shape[0],len(self.channel_pairs),signal.shape[2]))
        print(signal.shape[0])
        for k,ch_pair in enumerate(self.channel_pairs):
            ch1 = ch_pair[0]
            ch2 = ch_pair[1]
            for session in np.arange(signal.shape[0]):
                for trial in np.arange(signal.shape[2]):
                    X = signal[session,ch1,trial,self.time_window[0]:self.time_window[1]]
                    Y = signal[session,ch2,trial,self.time_window[0]:self.time_window[1]]
                    DPLI[session,k,trial] = self.compute_dpli_band(signal_x=X,signal_y=Y,fs=self.fs,lowcut_x = self.lowcut_x,highcut_x = self.highcut_x,lowcut_y=self.lowcut_y,highcut_y=self.highcut_y)
        return DPLI
    def butter_bandpass(self,lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return b, a

    def bandpass_filter(self,data, lowcut, highcut, fs, order=4):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        return filtfilt(b, a, data)

    def compute_dpli_band(self,signal_x, signal_y, fs, lowcut_x, highcut_x,lowcut_y, highcut_y):
        # Step 1: filter each signal into precise band
        x_f = self.bandpass_filter(signal_x, lowcut_x, highcut_x, fs)
        y_f = self.bandpass_filter(signal_y, lowcut_y, highcut_y, fs)

        # Step 2: compute phases via Hilbert
        ph_x = np.angle(hilbert(x_f))
        ph_y = np.angle(hilbert(y_f))

        # Step 3: wrap phase differences
        delta = np.angle(np.exp(1j * (ph_x - ph_y)))

        # Step 4: compute dPLI
        return np.mean(delta > 0)


@register("features","Complexity")
class Complexity(FeatureFunction):
    def __init__(self,complexity_type: str, arguments: dict, channels : List[int],time_window: Tuple=(0,50)):
        self.complexity_type = complexity_type
        self.complexity_types = {"SampEn": self.SampEn, "FuzzyEn": self.FuzEn, "LZC": self.LZC}
        self.channels = channels
        self.argument = arguments
        self.time_window = time_window
        assert complexity_type in self.complexity_types.keys(), f"Complexity type {complexity_type} not recognized. Available types: {list(self.complexity_types.keys())}"
        self.complexity_types_args = {"SampEn": ["m", "r", "tau"], "FuzzyEn": ["m", "r", "n", "tau"], "LZC": ["num_levels"]}
        assert set(self.argument.keys()) == set(self.complexity_types_args[complexity_type]), f"Arguments for {complexity_type} must be {self.complexity_types_args[complexity_type]}"
   

    def compute(self, signal, **kwargs):
        Complexity = np.zeros((signal.shape[0],len(self.channels),signal.shape[2]))
        print(signal.shape[0])
        for k,ch in enumerate(self.channels):
            for session in np.arange(signal.shape[0]):
                for trial in np.arange(signal.shape[2]):
                        X = signal[session,k,trial,self.time_window[0]:self.time_window[1]]
                        Complexity[session,k,trial] = self.complexity_types[self.complexity_type](x=X,argument=self.argument)
        return Complexity 
    def SampEn(self,x,argument):
        """
        Sample Entropy (SampEn) of a univariate signal x.

        Parameters
        ----------
        x : 1D array
            Input signal
        m : int
            Embedding dimension
        r : float
            Threshold (commonly 0.15 * std(x))
        tau : int, optional
            Time lag (default = 1)

        Returns
        -------
        Out_SampEn : float
            The Sample Entropy of x
        P : list
            [total template matches of length m, total forward matches of length m+1]
        """
        m = argument["m"]
        tau = argument["tau"]
        r = argument["r"]

        if tau > 1:
            x = x[::tau]  # downsample

        N = len(x)
        P = np.zeros(2)
        xMat = np.zeros((m+1, N-m))

        for i in range(m+1):
            xMat[i, :] = x[i:N-m+i]

        for k in range(m, m+2):
            count = np.zeros(N-m)
            tempMat = xMat[:k, :]

            for i in range(N-k):
                # Chebyshev distance (max abs difference)
                dist = np.max(np.abs(tempMat[:, i+1:N-m] - tempMat[:, [i]]), axis=0)
                D = (dist < r).astype(int)
                count[i] = np.sum(D) / (N-m)

            P[k-m] = np.sum(count) / (N-m)

        Out_SampEn = np.log(P[0] / P[1])
        return Out_SampEn

    def FuzEn(self, x, argument):
        """
        Fuzzy Entropy (FuzEn) of a univariate signal x.

        Parameters
        ----------
        x : 1D array-like
            Input signal
        m : int
            Embedding dimension
        r : float
            Threshold (commonly 0.15 * std(x))
        n : float, optional
            Fuzzy power (default=2)
        tau : int, optional
            Time lag (default=1)

        Returns
        -------
        Out_FuzEn : float
            The Fuzzy Entropy of x
        P : list
            [global quantity in dimension m, global quantity in dimension m+1]
        """
        m = argument["m"]
        tau = argument["tau"]
        r = argument["r"]
        n = argument["n"]   

        x = np.asarray(x, dtype=float)

        if tau > 1:
            x = x[::tau]  # downsample

        N = len(x)
        P = np.zeros(2)
        xMat = np.zeros((m+1, N-m))

        # Construct template matrix
        for i in range(m+1):
            xMat[i, :] = x[i:N-m+i]

        # Loop for m and m+1
        for k in range(m, m+2):
            count = np.zeros(N-m)
            tempMat = xMat[:k, :]

            for i in range(N-k):
                # Chebyshev distance (max abs difference)
                dist = np.max(np.abs(tempMat[:, i+1:N-m] - tempMat[:, [i]]), axis=0)
                # Fuzzy membership function
                DF = np.exp(-(dist**n) / r)
                count[i] = np.sum(DF) / (N-m-1)

            P[k-m] = np.sum(count) / (N-m)

        Out_FuzEn = np.log(P[0] / P[1])
        return Out_FuzEn
    def LZC(x, argument):
        """
        Lempel-Ziv complexity of a univariate signal.

        Parameters
        ----------
        signal : 1D array-like
            Input signal
        num_levels : int
            Quantization levels (2 or 3)

        Returns
        -------
        out : float
            Lempel-Ziv complexity
        P : ndarray
            Quantized symbolic sequence
        """
        num_levels = argument["num_levels"]
        signal = x
        signal = np.asarray(signal, dtype=float)
        med = np.median(signal)

        # Quantization step
        if num_levels == 2:
            P = ((np.sign(signal - med) + 1) / 2).astype(int)
        elif num_levels == 3:
            P = np.zeros_like(signal, dtype=int)
            max_val = np.abs(np.max(signal))
            P[signal >= med + max_val / 16] = 2
            P[signal <= med - max_val / 16] = 0
            mask = (signal > med - max_val / 16) & (signal < med + max_val / 16)
            P[mask] = 1
        else:
            raise ValueError("num_levels must be 2 or 3")

        c = 2
        terminate = False
        r = 1
        i = 1

        while not terminate:
            S = P[:r]
            Q = P[r:r+i]

            # Check if Q exists in S+Q (excluding last symbol)
            concat = np.concatenate([S, Q])
            found = False
            if len(Q) > 0:
                for j in range(len(concat) - len(Q)):
                    if np.array_equal(concat[j:j+len(Q)], Q):
                        found = True
                        break

            if not found:
                c += 1
                r = r + i
                i = 1
            else:
                i += 1

            if r + i == len(P):
                terminate = True

        out = c * np.log2(len(P)) / len(P)
        return out


@register("features","averageout")# Test
class averageout(FeatureFunction):
    def __init__(self, dim: int):
        self.dim = dim
    def compute(self, signal, **kwargs) -> np.ndarray:
        return np.mean(signal, axis=self.dim)



#### Features for time_dynamics
@register("features", "WindowedFeature")
class WindowedFeature(FeatureFunction):
    """
    Apply a registered feature (e.g. ComplexityBase) over moving windows 
    along the samples axis, with two paradigms:
    
    Paradigm 1 ('per_epoch'):
        Compute the feature for each epoch and window separately, 
        then average across epochs.
        Output shape: (n_sessions, n_channels, n_windows)
        
    Paradigm 2 ('average_epochs'):
        First average across epochs, then compute the feature 
        over windows on the averaged signal.
        Output shape: (n_sessions, n_channels, n_windows)
    """

    def __init__(self, 
                 feature_name: str,
                 feature_args: dict,
                 window_size: int,
                 overlap: float = 0.5,
                 paradigm: str = "per_epoch"):
        assert paradigm in ["per_epoch", "average_epochs"], \
            "paradigm must be 'per_epoch' or 'average_epochs'"
        assert 0 <= overlap and overlap < 1, "overlap must be in [0, 1)"
        
        self.feature_name = feature_name
        self.feature_args = feature_args
        self.window_size = window_size
        self.overlap = overlap
        self.paradigm = paradigm
       
        # Retrieve registered feature constructor
        self.feature = REGISTRIES["features"][feature_name](**feature_args)

    def compute(self, signal: np.ndarray, **kwargs):
        """
        Parameters
        ----------
        signal : np.ndarray
            Expected shape: (n_sessions, n_channels, n_epochs, n_samples)
        Returns
        -------
        np.ndarray
            Shape: (n_sessions, n_channels, n_windows)
        """
        n_sessions, n_channels, n_epochs, n_samples = signal.shape
        step = int(self.window_size * (1 - self.overlap))
        n_windows = max(1, (n_samples - self.window_size) // step + 1)

        # Prepare output
        out = np.zeros((n_sessions, n_channels, n_windows))

        for s in range(n_sessions):
            for ch in range(n_channels):
                # Extract signal across epochs for this channel
                x = signal[s, ch, :, :]  # shape: (n_epochs, n_samples)

                if self.paradigm == "per_epoch":
                    # Compute feature for each epoch & window, then average
                    feat_epochs = []
                    for ep in range(n_epochs):
                        feat_ep = self._compute_over_windows(x[ep])
                        feat_epochs.append(feat_ep)
                    out[s, ch, :] = np.nanmean(np.vstack(feat_epochs), axis=0)

                elif self.paradigm == "average_epochs":
                    # Average first across epochs, then compute over windows
                    avg_signal = np.nanmean(x, axis=0)
                    out[s, ch, :] = self._compute_over_windows(avg_signal)

        return out

    def _compute_over_windows(self, x: np.ndarray) -> np.ndarray:
        """Apply the wrapped feature over all windows of a single 1D signal."""
        step = int(self.window_size * (1 - self.overlap))
        n_windows = max(1, (len(x) - self.window_size) // step + 1)
        feats = np.zeros(n_windows)

        for w in range(n_windows):
            start = w * step
            end = start + self.window_size
            if end > len(x):
                break
            window = x[start:end]
            feats[w] = self.feature.compute(window)
        return feats


@register("features", "WindowedFeatureParallel")
class WindowedFeature(FeatureFunction):
    """
    Efficient window-based feature wrapper with optional parallel computation.
    
    Applies a registered feature (e.g. ComplexityBase) over moving windows 
    along the samples axis, using vectorized sliding windows and optional parallelism.
    """

    def __init__(self, 
                 feature_name: str,
                 feature_args: dict,
                 window_size: int,
                 overlap: float = 0.5,
                 paradigm: str = "per_epoch",
                 n_jobs: int = -1):
        assert paradigm in ["per_epoch", "average_epochs"], \
            "paradigm must be 'per_epoch' or 'average_epochs'"
        assert 0 <= overlap and overlap < 1, "overlap must be in [0, 1)"
        
        self.feature_name = feature_name
        self.feature_args = feature_args
        self.window_size = window_size
        self.overlap = overlap
        self.paradigm = paradigm
        self.n_jobs = n_jobs

        # Retrieve registered feature constructor
        self.feature = REGISTRIES["features"][feature_name](**feature_args)

    def compute(self, signal: np.ndarray, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            Shape (n_sessions, n_channels, n_epochs, n_samples)
        
        Returns
        -------
        np.ndarray
            Shape (n_sessions, n_channels, n_windows)
        """
        data = signal
        n_sessions, n_channels, n_epochs, n_samples = data.shape
        step = int(self.window_size * (1 - self.overlap))
        n_windows = max(1, (n_samples - self.window_size) // step + 1)
        
        # Define computation for one (session, channel)
        def process_channel(session_idx, ch_idx):
            x = data[session_idx, ch_idx]  # shape (n_epochs, n_samples)

            if self.paradigm == "per_epoch":
                # Compute feature for each epoch → average across epochs
                feats_epochs = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._compute_over_windows)(x[ep]) 
                    for ep in range(n_epochs)
                )
                return np.nanmean(feats_epochs, axis=0)

            else:  # average_epochs
                x_mean = np.nanmean(x, axis=0)
                return self._compute_over_windows(x_mean)

        # Parallel over session × channel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_channel)(s, ch)
            for s in range(n_sessions)
            for ch in range(n_channels)
        )

        # Reshape results
        out = np.array(results).reshape(n_sessions, n_channels, n_windows)
        return out

    def _compute_over_windows(self, x: np.ndarray):
        """Vectorized window slicing and feature computation."""
        step = int(self.window_size * (1 - self.overlap))
        # Get all sliding windows efficiently
        windows = sliding_window_view(x, self.window_size)[::step]
        # Apply feature on each window in parallel (short signals -> lightweight)
        feats = [self.feature.compute(win) for win in windows]
        return np.array(feats)



@register("features", "EventWindowedFeature")
class EventWindowedFeature(FeatureFunction):
    """
    Event-based time-dynamic feature computation.
    """

    def __init__(self,
                 feature_name: str,
                 feature_args: dict,
                 event_type,
                 window_size: int,
                 channels: list,
                 overlap: float = 0.5,
                 paradigm: str = "per_epoch"):

        assert paradigm in ["per_epoch", "average_epochs"], \
            "paradigm must be 'per_epoch' or 'average_epochs'"
        assert 0 <= overlap < 1, "overlap must be in [0,1)"

        self.feature = REGISTRIES["features"][feature_name](**feature_args)
        self.event_type = event_type if isinstance(event_type, (list, tuple)) else [event_type]
        self.window_size = window_size
        self.overlap = overlap
        self.paradigm = paradigm
        self.channels = channels

        # choose compute strategy once
        if paradigm == "per_epoch":
            self._compute_strategy = self._compute_per_epoch
        else:
            self._compute_strategy = self._compute_average_epochs

    def compute(self, signal=None, events=None, **kwargs):
        print("linear")
        """
        signal : (n_sessions, n_channels, n_epochs, n_samples)
        events : (n_sessions, n_epochs)
        returns (n_sessions, n_channels, n_windows)
        """
        if signal is None:
            raise ValueError("EventWindowedFeature requires `signal`")
        if events is None:
            raise ValueError("EventWindowedFeature requires `events` matrix")

        return self._compute_strategy(signal, events)

    # --------------------------------------------------------------
    # Paradigm 1: Compute for each epoch, then average across epochs
    # --------------------------------------------------------------
    def _compute_per_epoch(self, signal, events):
        n_sessions, n_channels, _, n_samples = signal.shape
        step = int(self.window_size * (1 - self.overlap))
        n_windows = max(1, (n_samples - self.window_size) // step + 1)

        out = np.zeros((n_sessions, len(self.channels), n_windows))

        for s in range(n_sessions):
            ep_mask = np.isin(events[s], self.event_type)
            selected = signal[s, :, ep_mask, :]  # (n_channels, n_selected_epochs, n_samples)

            if selected.shape[1] == 0:
                out[s, :, :] = np.nan
                continue

            for ch in self.channels:
                epochs = selected[ch]  # (n_selected_epochs, n_samples)
                feats = [self._compute_over_windows(ep) for ep in epochs]
                out[s, ch] = np.nanmean(np.vstack(feats), axis=0)

        return out

    # --------------------------------------------------------------
    # Paradigm 2: Average selected epochs first, then compute
    # --------------------------------------------------------------
    def _compute_average_epochs(self, signal, events):
        n_sessions, _, _, n_samples = signal.shape
        step = int(self.window_size * (1 - self.overlap))
        n_windows = max(1, (n_samples - self.window_size) // step + 1)

        out = np.zeros((n_sessions, len(self.channels), n_windows))

        for s in range(n_sessions):
            ep_mask = np.isin(events[s], self.event_type)
            selected = signal[s, :, ep_mask, :]  # (n_channels, n_selected_epochs, n_samples)

            if selected.shape[1] == 0:
                out[s, :, :] = np.nan
                continue

            for i,ch in enumerate(self.channels):
                avg_x = np.nanmean(selected[ch], axis=0)
                out[s, i] = self._compute_over_windows(avg_x)

        return out

    # --------------------------------------------------------------
    # Window helper
    # --------------------------------------------------------------
    def _compute_over_windows(self, x: np.ndarray):
        step = int(self.window_size * (1 - self.overlap))
        wins = sliding_window_view(x, self.window_size)[::step]
        return np.array([self.feature.compute(win) for win in wins])




@register("features", "EventWindowedFeatureParallel")
class EventWindowedFeature(FeatureFunction):
    def __init__(self,
                 feature_name: str,
                 feature_args: dict,
                 window_size: int,
                 event_type: int = None,
                 overlap: float = 0.5,
                 paradigm: str = "per_epoch",
                 n_jobs: int = -1):

        assert paradigm in ["per_epoch", "average_epochs"], \
            "paradigm must be 'per_epoch' or 'average_epochs'"
        assert 0 <= overlap < 1, "overlap must be in [0,1)"

        self.feature = REGISTRIES["features"][feature_name](**feature_args)
        self.event_type = event_type if isinstance(event_type, (list, tuple)) else [event_type]
        self.window_size = window_size
        self.overlap = overlap
        self.paradigm = paradigm
        self.n_jobs = n_jobs

        # choose strategy once
        self._compute_strategy = (
            self._compute_per_epoch if paradigm == "per_epoch"
            else self._compute_average_epochs
        )

    def compute(self, signal=None, events=None, **kwargs):
        print("parallel")
        if signal is None:
            raise ValueError("EventWindowedFeature requires `signal`")
        if events is None:
            raise ValueError("EventWindowedFeature requires `events` matrix")
        return self._compute_strategy(signal, events)

    # --------------------------------------------------------------
    # Parallel helper for per-epoch paradigm
    # --------------------------------------------------------------
    def _compute_per_epoch(self, signal, events):
        n_sessions, n_channels, _, n_samples = signal.shape
        step = int(self.window_size * (1 - self.overlap))
        n_windows = max(1, (n_samples - self.window_size) // step + 1)
        out = np.zeros((n_sessions, n_channels, n_windows))

        def process_session_channel(s, ch):
            ep_mask = (np.isin(events[s], self.event_type) if self.event_type != None else np.ones_like(events[s]))
            selected = signal[s, ch, ep_mask, :]  # (n_selected_epochs, n_samples)
            if selected.shape[0] == 0:
                return np.full(n_windows, np.nan)
            feats = [self._compute_over_windows(ep) for ep in selected]
            return np.nanmean(np.vstack(feats), axis=0)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_session_channel)(s, ch)
            for s in range(n_sessions)
            for ch in range(n_channels)
        )

        return np.array(results).reshape(n_sessions, n_channels, n_windows)

    # --------------------------------------------------------------
    # Parallel helper for average-epochs paradigm
    # --------------------------------------------------------------
    def _compute_average_epochs(self, signal, events):
        n_sessions, n_channels, _, n_samples = signal.shape
        step = int(self.window_size * (1 - self.overlap))
        n_windows = max(1, (n_samples - self.window_size) // step + 1)
        out = np.zeros((n_sessions, n_channels, n_windows))

        def process_session_channel(s, ch):
            ep_mask = np.isin(events[s], self.event_type)
            selected = signal[s, ch, ep_mask, :]  # (n_selected_epochs, n_samples)
            if selected.shape[0] == 0:
                return np.full(n_windows, np.nan)
            avg_x = np.nanmean(selected, axis=0)
            return self._compute_over_windows(avg_x)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_session_channel)(s, ch)
            for s in range(n_sessions)
            for ch in range(n_channels)
        )

        return np.array(results).reshape(n_sessions, n_channels, n_windows)

    # --------------------------------------------------------------
    # Window helper (efficient)
    # --------------------------------------------------------------
    def _compute_over_windows(self, x: np.ndarray):
        step = int(self.window_size * (1 - self.overlap))
        windows = sliding_window_view(x, self.window_size)[::step]
        return np.array([self.feature.compute(win) for win in windows])
        # try:
        #     return np.array([self.feature.compute(win) for win in windows])
        # except Warning as w:
        #     import pdb; pdb.set_trace()
        #     return np.nan
        


@register("features", "EventWindowedPairFeature")
class EventWindowedPairFeature(FeatureFunction):
    """
    Event-based time-dynamic feature computation for pairwise channel features.
    The base feature must accept two signals per window: feature.compute(x_win, y_win)
    """

    def __init__(self,
                 feature_name: str,
                 feature_args: dict,
                 channel_pairs: list,
                 event_type,
                 window_size: int,
                 overlap: float = 0.5,
                 paradigm: str = "per_epoch"):

        assert paradigm in ["per_epoch", "average_epochs"], \
            "paradigm must be 'per_epoch' or 'average_epochs'"
        assert 0 <= overlap < 1, "overlap must be in [0,1)"

        self.feature = REGISTRIES["features"][feature_name](**feature_args)
        self.channel_pairs = channel_pairs     # list of (ch1, ch2) pairs
        self.event_type = event_type if isinstance(event_type, (list, tuple)) else [event_type]
        self.window_size = window_size
        self.overlap = overlap
        self.paradigm = paradigm

        self._compute_strategy = (
            self._compute_per_epoch if paradigm == "per_epoch"
            else self._compute_average_epochs
        )

    def compute(self, signal=None, events=None, **kwargs):
        """
        signal : (n_sessions, n_channels, n_epochs, n_samples)
        events : (n_sessions, n_epochs)
        returns (n_sessions, n_pairs, n_windows)
        """
        if signal is None:
            raise ValueError("EventWindowedPairFeature requires `signal`")
        if events is None:
            raise ValueError("EventWindowedPairFeature requires `events`")

        return self._compute_strategy(signal, events)

    # --------------------------------------------------------------
    # Paradigm 1: Compute per epoch, then average across epochs
    # --------------------------------------------------------------
    def _compute_per_epoch(self, signal, events):
        n_sessions, n_channels, _, n_samples = signal.shape
        n_pairs = len(self.channel_pairs)
        step = int(self.window_size * (1 - self.overlap))
        n_windows = max(1, (n_samples - self.window_size) // step + 1)

        out = np.zeros((n_sessions, n_pairs, n_windows))

        for s in range(n_sessions):
            ep_mask = np.isin(events[s], self.event_type)
            selected = signal[s, :, ep_mask, :]  # (n_channels, n_selected_epochs, n_samples)
            

            assert selected.shape[1] != 0
            # if selected.shape[1] == 0:
            #     out[s, :, :] = np.nan
            #     continue

            for idx, (ch1, ch2) in enumerate(self.channel_pairs):
                epochs_x = selected[ch1]  # (n_selected_epochs, n_samples)
                epochs_y = selected[ch2]

                feats = [self._compute_over_windows(epx, epy) for epx, epy in zip(epochs_x, epochs_y)]
                out[s, idx] = np.nanmean(np.vstack(feats), axis=0)

        return out

    # --------------------------------------------------------------
    # Paradigm 2: Average epochs first, then compute
    # --------------------------------------------------------------
    def _compute_average_epochs(self, signal, events):
        n_sessions, n_channels, _, n_samples = signal.shape
        n_pairs = len(self.channel_pairs)
        step = int(self.window_size * (1 - self.overlap))
        n_windows = max(1, (n_samples - self.window_size) // step + 1)

        out = np.zeros((n_sessions, n_pairs, n_windows))

        for s in range(n_sessions):
            ep_mask = np.isin(events[s], self.event_type)
            selected = signal[s, :, ep_mask, :]  # (n_channels, n_selected_epochs, n_samples)

            if selected.shape[1] == 0:
                out[s, :, :] = np.nan
                continue

            for idx, (ch1, ch2) in enumerate(self.channel_pairs):
                avg_x = np.nanmean(selected[ch1], axis=0)
                avg_y = np.nanmean(selected[ch2], axis=0)
                out[s, idx] = self._compute_over_windows(avg_x, avg_y)

        return out

    # --------------------------------------------------------------
    # Window helper for pairwise features
    # --------------------------------------------------------------
    def _compute_over_windows(self, x: np.ndarray, y: np.ndarray):
        step = int(self.window_size * (1 - self.overlap))
        X_wins = sliding_window_view(x, self.window_size)[::step]
        Y_wins = sliding_window_view(y, self.window_size)[::step]
        return np.array([self.feature.compute(wx, wy) for wx, wy in zip(X_wins, Y_wins)])





@register("features", "EventWindowedPairFeatureParallel")
class EventWindowedPairFeatureParallel(FeatureFunction):
    """
    Event-based time-dynamic feature computation for pairwise channel features.
    The base feature must accept two signals per window:
        feature.compute(x_window, y_window)

    Input:
        signal : (n_sessions, n_channels, n_epochs, n_samples)
        events : (n_sessions, n_epochs)
        channel_pairs : list of (ch1, ch2) index tuples

    Output:
        (n_sessions, n_pairs, n_windows)
    """

    def __init__(self,
                 feature_name: str,
                 feature_args: dict,
                 channel_pairs: list,
                 event_type,
                 window_size: int,
                 overlap: float = 0.5,
                 paradigm: str = "per_epoch"):

        assert paradigm in ["per_epoch", "average_epochs"], \
            "paradigm must be 'per_epoch' or 'average_epochs'"
        assert 0 <= overlap < 1, "overlap must be in [0,1)"

        self.feature = REGISTRIES["features"][feature_name](**feature_args)
        self.channel_pairs = channel_pairs
        self.event_type = event_type if isinstance(event_type, (list, tuple)) else [event_type]
        self.window_size = window_size
        self.overlap = overlap
        self.paradigm = paradigm

        self._compute_strategy = (
            self._compute_per_epoch if paradigm == "per_epoch"
            else self._compute_average_epochs
        )

    def compute(self, signal=None, events=None, **kwargs):
        print(signal.shape)
        if signal is None:
            raise ValueError("EventWindowedPairFeature requires `signal`")
        if events is None:
            raise ValueError("EventWindowedPairFeature requires `events`")

        return self._compute_strategy(signal, events)

    # ==============================================================
    # Shared helper: compute windowed features on 1D signals
    # ==============================================================
    def _compute_over_windows(self, x, y):
        step = self._step
        X = sliding_window_view(x, self.window_size)[::step]
        Y = sliding_window_view(y, self.window_size)[::step]
        return np.array([self.feature.compute(wx, wy) for wx, wy in zip(X, Y)])

    # ==============================================================
    # Paradigm 1: Compute per epoch, then average across epochs
    # ==============================================================
    def _compute_per_epoch(self, signal, events):
        n_sessions, _, _, n_samples = signal.shape
        n_pairs = len(self.channel_pairs)

        self._step = int(self.window_size * (1 - self.overlap))
        self._n_windows = max(1, (n_samples - self.window_size) // self._step + 1)

        def process_session(s):
            ep_mask = np.isin(events[s], self.event_type)
            selected = signal[s, :, ep_mask, :]   # (channels, epochs, samples)

            if selected.shape[1] == 0:
                return np.full((n_pairs, self._n_windows), np.nan)

            session_out = np.zeros((n_pairs, self._n_windows))

            for idx, (ch1, ch2) in enumerate(self.channel_pairs):
                ex = selected[ch1]       # (epochs, samples)
                ey = selected[ch2]
                feats = [self._compute_over_windows(x, y) for x, y in zip(ex, ey)]
                session_out[idx] = np.nanmean(np.vstack(feats), axis=0)

            return session_out

        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(process_session)(s) for s in range(n_sessions)
        )

        return np.stack(results, axis=0)

    # ==============================================================
    # Paradigm 2: Average selected epochs first, then compute
    # ==============================================================
    def _compute_average_epochs(self, signal, events):
        n_sessions, _, _, n_samples = signal.shape
        n_pairs = len(self.channel_pairs)

        self._step = int(self.window_size * (1 - self.overlap))
        self._n_windows = max(1, (n_samples - self.window_size) // self._step + 1)

        def process_session(s):
            ep_mask = np.isin(events[s], self.event_type)
            selected = signal[s, :, ep_mask, :]   # (channels, epochs, samples)

            if selected.shape[1] == 0:
                return np.full((n_pairs, self._n_windows), np.nan)

            avg = np.nanmean(selected, axis=1)     # (channels, samples)

            session_out = np.zeros((n_pairs, self._n_windows))
            for idx, (ch1, ch2) in enumerate(self.channel_pairs):
                session_out[idx] = self._compute_over_windows(avg[ch1], avg[ch2])

            return session_out

        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(process_session)(s) for s in range(n_sessions)
        )

        return np.stack(results, axis=0)




@register("features","cross-correlation-di-fc")
# Cross-correlation functional connectivity
# Implemented from the following article: 
# Cross-correlation of instantaneous amplitudes of field potential oscillations: a straightforward method to estimate the directionality and lag between brain areas
class CrossCorrelationConnectivity(FeatureFunction):
    def __init__(self, fs, channel_pairs: List[Tuple], band=(7,12),maxlag_sec=0.1,n_boot=1000,alpha=0.05, time_window: Tuple=(0,50)):
        super().__init__()
        self.band = band
        self.maxlag_sec = maxlag_sec
        self.n_boot = n_boot
        self.alpha = alpha
        self.channel_pairs = channel_pairs
        self.time_window = time_window
        self.fs = fs

    def compute(self, signal, **kwargs):
        CCDFC = np.zeros((signal.shape[0],len(self.channel_pairs),signal.shape[2]))
        print(signal.shape[0])
        for k,ch_pair in enumerate(self.channel_pairs):
            ch1 = ch_pair[0]
            ch2 = ch_pair[1]
            for session in np.arange(signal.shape[0]):
                for trial in np.arange(signal.shape[2]):
                    X = signal[session,ch1,trial,self.time_window[0]:self.time_window[1]]
                    Y = signal[session,ch2,trial,self.time_window[0]:self.time_window[1]]
                    
                    result = self.amplitude_xcorr_directionality(x=X,y=Y,fs=self.fs,band = self.band, 
                                                                                maxlag_sec=self.maxlag_sec, alpha = self.alpha, n_boot = self.n_boot)
                    if result['significant']:
                        if result['x_leads_y']:
                            CCDFC[session,k,trial]  = result["peak_val"]
                        else:
                            CCDFC[session,k,trial]  = -result["peak_val"]   
        return CCDFC 
            
    def amplitude_xcorr_directionality(self,x, y, fs,
                                   band=(7,12),
                                   maxlag_sec=0.1,
                                   n_boot=1000,
                                   alpha=0.05):
        """
        Estimate directionality from x -> y using the amplitude cross-correlation
        method of Adhikari et al. (2010).

        Parameters
        ----------
        x, y : 1D numpy arrays (same length)
            Raw time series (e.g. LFP traces).
        fs : float
            Sampling frequency in Hz.
        band : tuple (low_hz, high_hz)
            Bandpass range in Hz (default (7,12) as in the paper).
        maxlag_sec : float
            Look for cross-correlation peaks within +/- this many seconds (default 0.1s).
        n_boot : int
            Number of bootstrap circular shifts to generate null distribution (default 1000).
        alpha : float
            Significance level for bootstrap (default 0.05).

        Returns
        -------
        result : dict with keys
        'lag_s'      : lag in seconds at which the amplitude cross-correlation peaks.
                        (See sign convention note below.)
        'peak_val'   : cross-correlation value at the peak (unnormalized).
        'p_value'    : empirical p-value from bootstrap (fraction of boot peaks >= original).
        'significant': boolean, True if p_value < alpha.
        'lags'       : array of lag values (seconds) for the returned xcorr array.
        'xcorr'      : cross-correlation array (restricted to +/- maxlag_sec).
        'env_x','env_y': instantaneous amplitude envelopes (abs(Hilbert(filtered signal))).
        'x_leads_y'  : boolean (True if lag_s < 0 according to the paper's convention).
        """
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        n = x.size
        # --- 1) Bandpass with an FIR like paper (order ≈ fs samples) ---
        nyq = fs / 2.0
        low = band[0] / nyq
        high = band[1] / nyq
        numtaps = int(round(fs))
        # adapt filter length to signal length
        numtaps = min(int(fs), int(len(x) // 3))
        if numtaps % 2 == 0:
            numtaps += 1
        if numtaps < 5:
            numtaps = 5  # minimum to avoid degenerate filter
        taps = firwin(numtaps, [low, high], pass_zero=False, window='hamming')
        x_f = filtfilt(taps, [1.0], x)
        y_f = filtfilt(taps, [1.0], y)

        # --- 2) Instantaneous amplitude (Hilbert) ---
        env_x = np.abs(hilbert(x_f))
        env_y = np.abs(hilbert(y_f))

        # subtract mean (DC) before cross-correlation (paper step)
        ex = env_x - np.mean(env_x)
        ey = env_y - np.mean(env_y)

        # --- 3) Cross-correlation (we correlate ey with ex) ---
        full = correlate(ey, ex, mode='full')
        lags_full = np.arange(-n+1, n)
        maxlag_samples = int(round(maxlag_sec * fs))

        center = len(full)//2
        idx0 = center - maxlag_samples
        idx1 = center + maxlag_samples + 1
        xcorr = full[idx0:idx1]
        lags = lags_full[idx0:idx1] / float(fs)

        peak_idx = np.argmax(xcorr)
        peak_val = float(xcorr[peak_idx])
        lag_s = float(lags[peak_idx])

        # --- 4) Bootstrap null with random circular shifts (5-10 s as in paper) ---
        minshift = int(round(5 * fs))
        maxshift = int(round(10 * fs))
        if n <= minshift:
            minshift = 1
        if n <= maxshift:
            maxshift = max(1, n // 2)

        rng = np.random.default_rng()
        boot_peaks = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            shift = rng.integers(minshift, maxshift+1)
            ey_shift = np.roll(ey, shift)
            fullb = correlate(ey_shift, ex, mode='full')
            xb = fullb[idx0:idx1]
            boot_peaks[i] = np.max(xb)

        # empirical (one-sided) p-value: fraction of boot peaks >= observed peak
        p_val = (np.sum(boot_peaks >= peak_val) + 1) / (n_boot + 1)
        significant = (p_val < alpha)

        return {
            'lag_s': lag_s,
            'peak_val': peak_val,
            'p_value': p_val,
            'significant': significant,
            'lags': lags,
            'xcorr': xcorr,
            'env_x': env_x,
            'env_y': env_y,
            'x_leads_y': (lag_s < 0)
        }
