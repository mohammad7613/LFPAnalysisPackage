# features/power_band.py
# lfp_analysis/business_logic/spectral.py
"""Spectral feature implementations (example: band power)."""
from typing import Tuple
import numpy as np
from .base import FeatureFunction
from scipy.signal import welch
from npeet import entropy_estimators as ee
from typing import List, Tuple
from lfp_analysis.registery import register
from scipy.signal import butter, filtfilt, hilbert

@register("features","TE")
class TE(FeatureFunction):
    def __init__(self,d_x: int, d_y: int, w_x: int, w_y: int, channel_pairs: List[Tuple], time_window: Tuple):
        self.d_x = d_x
        self.d_y = d_y
        self.w_x = w_x
        self.w_y = w_y
        self.channel_pairs = channel_pairs
        self.time_window = time_window


    def compute(self, signal: np.ndarray, **kwargs) -> np.ndarray:
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
    def compute(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Compute bandpower.
        If `signal` has shape (n_samples,) -> return float.
        If `signal` has shape (n_sessions, n_channels, n_epochs, n_samples) ->
        return (n_sessions, n_epochs) averaged across channels.
        """
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

                