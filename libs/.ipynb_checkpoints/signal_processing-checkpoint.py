import sys
import copy
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from scipy.signal import welch
import random
import antropy as ant
from scipy.fft import fft, fftfreq

class WindowIIRNotchFilter():
    '''
    Class for real time IIR notch filtering.
    '''

    def __init__(self, w0, Q, fs):
        '''
        Arguments:
            w0: Center frequency of notch filter.
            Q: Quality factory of notch filter.
            fs: Sampling rate of signal that filtering will be performed on.
        '''
        self.w0 = w0
        self.Q = Q
        self.fs = fs
        self.initialize_filter_params()

    def initialize_filter_params(self):
        '''
        Initialize IIR notch filter parameters.
        '''
        self.b, self.a = scipy.signal.iirnotch(self.w0, self.Q, self.fs)
        self.z = scipy.signal.lfilter_zi(self.b, self.a)

    def filter_data(self, x):
        '''
        Apply notch filter to signal sample x.

        input:
            x: Window of signal data
        output:
            result: Filtered signal data
        '''
        result, zf = scipy.signal.lfilter(self.b, self.a, x, -1, self.z)
        self.z = zf
        return np.array(result)


class DCBlockingFilter:
    '''
    Class for window-based time DC Blocking filtering.
    '''

    def __init__(self, alpha=0.99):
        '''
        Arguments:
            alpha: Adaptation time-constant for DC drift
        '''
        self.alpha = alpha
        self.initialize_filter_params()

    def initialize_filter_params(self):
        '''
        Initialize filter parameters.
        '''
        self.b = [1, -1]
        self.a = [1, -1 * self.alpha]
        self.zi = scipy.signal.lfilter_zi(self.b, self.a)

    def filter_data(self, x):
        '''
        Apply filter to signal sample x.

        input:
            x: Window of signal data
        output:
            result: Filtered signal data
        '''
        result, zf = scipy.signal.lfilter(self.b, self.a, x, zi=self.zi)
        self.zi = zf
        return np.array(result)


class WindowFilter():
    '''
    Sliding window filtering class for de-noising slow wave data in deep sleep epochs.
    '''

    def __init__(self, filters):
        '''
        Arguments:
            filters: list of RealTime filter objects
        '''
        self.filters = filters

    def initialize_filter_params(self):
        '''
        Initializes RealTime filter object parameters.
        '''
        for filt in self.filters:
            filt.initialize_filter_params()

    def filter_data(self, x):
        '''
        Apply RealTime filters to signal sample x.

        input:
            x: Window of signal data
        output:
            result: Filtered signal data
        '''
        for filt in self.filters:
            x = filt.filter_data(x)
        return x


class WindowButterBandpassFilter():
    '''
    Class for real time Butterworth Bandpass filtering.
    '''

    def __init__(self, order, low, high, fs):
        '''
        Arguments:
            order: Bandpass filter order.
            low: Lower cutoff frequency (Hz).
            high: Higher cutoff frequency (Hz).
            fs: Sampling rate of signal that filtering will be performed on.
        '''
        self.order = order
        self.low = low
        self.high = high
        self.fs = fs
        self.initialize_filter_params()

    def initialize_filter_params(self):
        '''
        Initialize filter parameters.
        '''
        self.b, self.a = scipy.signal.butter(self.order, [self.low, self.high], btype='band', fs=self.fs)
        self.z = scipy.signal.lfilter_zi(self.b, self.a)

    def filter_data(self, x):
        '''
        Apply bandpass filter to signal sample x.

        input:
            x: Window of signal data
        output:
            result: Filtered signal data
        '''
        x = np.reshape(x, (-1,))
        result, zf = scipy.signal.lfilter(self.b, self.a, x, zi=self.z)
        self.z = zf
        return np.array(result)
    
    


def simpson(y, x=None, dx=1.0):
    """
    Numerical integration using Simpson's rule.
    If the number of points is even, the last interval is handled using the trapezoidal rule.

    Parameters:
    - y: Array of function values.
    - x: Array of sample points (optional).
    - dx: Spacing between points (used if x is None).

    Returns:
    - Integral approximation.
    """
    y = np.asarray(y)
    n = len(y)
    if n < 2:
        raise ValueError("At least 2 points are required.")
    
    if x is None:
        x = np.arange(n) * dx
    else:
        x = np.asarray(x)
        if len(x) != n:
            raise ValueError("x and y must have the same length.")

    if n % 2 == 1:
        # Odd number of points: apply Simpson's rule directly
        h = (x[1:] - x[:-1]).mean()
        result = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
        result *= h / 3.0
    else:
        # Even number of points: apply Simpson's rule to first n-1, trapezoid to last interval
        result = simpson(y[:-1], x[:-1])  # recursive call on n-1 points
        trap = 0.5 * (x[-1] - x[-2]) * (y[-1] + y[-2])
        result += trap

    return result


def compute_z_ratio(datum, fs):
    '''
    Computes the z-ratio of a given signal segment.

    input:
        datum: segment of biopotential signal data
        fs: sampling rate of datum
    output:
        z-ratio, a measure of relative slow wave activity in the EEG signal
    '''
    fft_datum = np.abs(fft(datum))
    freqs = fftfreq(len(datum),1/fs)
    indice = np.bitwise_and(freqs<=(20), freqs>=0.5)
    fft_datum = fft_datum[indice]
    freqs = freqs[indice]
    total_pow = simpson(fft_datum,freqs)

    slow_indice = np.bitwise_and(freqs<=8, freqs>=0.5)
    slow_power = simpson(fft_datum[slow_indice],freqs[slow_indice])/(total_pow +1e-10)

    fast_indice = np.bitwise_and(freqs<=20, freqs>=8)
    fast_power = simpson(fft_datum[fast_indice],freqs[fast_indice])/(total_pow +1e-10)

    return (slow_power-fast_power)/(slow_power+fast_power +1e-10), slow_power/(fast_power +1e-10)

def eeg_features(window_data, fs=128, nperseg=128):
    # Welch PSD Calculation
    nperseg = min(len(window_data), nperseg)
    freqs, psd = welch(window_data, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd)

    # Frequency bands
    bands = {'delta': (0.5, 4),
             'theta': (4, 8),
             'alpha': (8, 13),
             'beta':  (13, 30),
             'gamma': (30, 50)}

    # Band power calculations
    band_power = {}
    for band, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs < high)
        band_power[band] = np.sum(psd[idx])
        
    band_power_merge ={
    'delta':band_power['delta'],
    'theta':band_power['theta'],
    'alpha': band_power['alpha'],
    'beta':band_power['beta'],
    'gamma':band_power['gamma']
    }

    # Normalized power
    norm_band_power = {b: p/(total_power + 1e-10)
    for b, p in band_power.items()}

    # Ratios
    ratios = {
        'theta_alpha': band_power['theta']/(band_power['alpha'] + 1e-10),
        'theta_beta': band_power['theta']/(band_power['beta'] + 1e-10),
        'theta_alpha_beta': band_power['theta']/(band_power['alpha'] + band_power['beta'] + 1e-10),
        'beta_alpha': band_power['beta']/(band_power['alpha'] + 1e-10),
        'beta_gamma': band_power['beta']/(band_power['gamma'] + 1e-10),
        'alpha_gamma': band_power['alpha']/(band_power['gamma'] + 1e-10)
    }


    spec_entropy = ant.spectral_entropy(window_data, sf=fs,
                                        method='welch', normalize=True)

    if np.isnan(spec_entropy):
        print("Warning: Input signal contains NaN or inf values.")
        spec_entropy = 1e10

    statistical_features = [np.std(window_data), np.max(window_data) - np.min(window_data)]

    ret0, ret1 = compute_z_ratio(window_data, fs)
    z_ratio_features = [ret0, ret1]


    # Concate all values
    features = np.array([
        *band_power_merge.values(),     # 5 features
        *norm_band_power.values(),      # 5 features
        *ratios.values(),               # 6 features
        *statistical_features,          # 3 features
        *z_ratio_features,              # 1 feature
        spec_entropy                    # 1 feature
    ])
    return features


def get_saccade_features(eog_data, fs):
    velocity = np.diff(eog_data)*fs
    acceleration = np.diff(velocity)*fs
    mean_vel = np.mean(velocity)
    max_vel = np.max(velocity)
    return [mean_vel, max_vel]

def eog_features(window_data, fs=128, nperseg=128):
    deflection = np.abs(window_data)
    max_deflection = np.max(deflection)
    percentile_60_deflection = np.percentile(deflection, 60)
    percentile_90_deflection = np.percentile(deflection, 90)
    
    # Welch PSD Calculation
    nperseg = min(len(window_data), nperseg)
    freqs, psd = welch(window_data, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd)

    # Frequency bands
    bands = {'low': (0.3, 4),
             'high': (4, 10)}
    
    # Band power calculations
    band_power = {}
    for band, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs < high)
        band_power[band] = np.sum(psd[idx])
        
    
    low_power_eog, high_power_eog = band_power['low'], band_power['high']

    features = []
    features += [max_deflection, percentile_60_deflection, percentile_90_deflection] # 3
    features += list(get_saccade_features(window_data, fs))  # 2 
    features += [low_power_eog, high_power_eog] # 2
    features += [np.max(window_data)-np.min(window_data)] # 1
    return np.asarray(features)



if __name__ == "__main__":
    fs = 244
    dc_filter = DCBlockingFilter(alpha=0.99)
    notch_60 = WindowIIRNotchFilter(60, 12, fs)
    notch_50 = WindowIIRNotchFilter(50, 5, fs)
    notch_25 = WindowIIRNotchFilter(25, 10, fs)
    bandpass_filter = WindowButterBandpassFilter(3, 1, 35, fs)

    eeg_filter = WindowFilter([dc_filter, notch_60, notch_50, notch_25, bandpass_filter])


    duration = 30  # seconds
    total_samples = fs * duration

    t = np.arange(total_samples) / fs
    raw_signal = ( 300 +
            20 * np.sin(2 * np.pi * 1 * t) +  # 1 Hz delta wave
            10 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz alpha wave
            10 * np.sin(2 * np.pi * 60 * t) +  # 60 Hz noise
            20 * np.random.randn(total_samples)  # random noise
    )

    window_len = fs
    filtered_signal = []

    for i in range(0, total_samples, window_len):
        window = raw_signal[i:i + window_len]
        if len(window) < window_len:
            break  # skip incomplete window at the end
        filtered_window = eeg_filter.filter_data(window)
        filtered_signal.extend(filtered_window)

    filtered_signal = np.array(filtered_signal)

    # Plot comparison
    plt.figure(figsize=(12, 5))
    plt.plot(t, raw_signal, label='Raw Signal', alpha=0.5)
    plt.plot(t[:len(filtered_signal)], filtered_signal, label='Filtered Signal', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("EEG Signal Filtering Example")
    plt.tight_layout()
    plt.show()