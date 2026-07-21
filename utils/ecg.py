"""
Self-contained ECG loading and preprocessing: the Butterworth/notch filter chain applied to the
raw PTB-XL recordings, R-peak detection, and R-centred single-beat extraction.

The filter chain reproduces the preprocessing the detectors were trained with, so an example
record loaded here reaches the model in exactly the layout it expects, namely time-major
(num_samples, num_leads).
"""
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch

from utils.config import DEFAULT_ECG_DIR, ECG_FILTERS, SAMPLING_RATE


def perform_shape_switch(a):
    """Transpose a 2D array, switching between lead-major and time-major layout."""
    a = np.asarray(a)
    dimx, dimy = a.shape
    output = np.zeros((dimy, dimx))

    for i in range(dimx):
        output[:, i] = a[i, :]

    return output


def derive_filter_values(filters):
    """
    Parse filter strings such as 'LP40Hz' or 'AC50Hz' into a {type: cutoff} dict. The
    parameterless filters 'BWR' (baseline wander removal) and 'BLA' (baseline adjustment) map
    to None.
    """
    filter_values = {}

    if filters is not None:
        for filterstring in filters:
            if filterstring in ['BWR', 'BLA']:
                filter_values[filterstring] = None
            else:
                f = filterstring[:2]
                v = float(filterstring[2:-2])
                filter_values[f] = v

    return filter_values


def butter_lowpass(cutoff, sample_rate, order=2):
    """Butterworth lowpass filter coefficients (b, a)."""
    nyq = 0.5 * sample_rate

    return butter(order, cutoff / nyq, btype='low', analog=False)


def butter_highpass(cutoff, sample_rate, order=2):
    """Butterworth highpass filter coefficients (b, a)."""
    nyq = 0.5 * sample_rate

    return butter(order, cutoff / nyq, btype='high', analog=False)


def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass'):
    """Apply a lowpass, highpass or notch filter to a single lead."""
    if filtertype == 'lowpass':
        b, a = butter_lowpass(cutoff, sample_rate, order=order)
    elif filtertype == 'highpass':
        b, a = butter_highpass(cutoff, sample_rate, order=order)
    elif filtertype == 'notch':
        b, a = iirnotch(cutoff, Q=0.05, fs=sample_rate)
    else:
        raise ValueError('Unknown filtertype "{}", available are: lowpass, highpass, notch'.format(filtertype))

    return filtfilt(b, a, data)


def remove_baseline_wander(data, sample_rate, cutoff=0.05):
    """Remove baseline wander with a narrow notch filter around the given low frequency."""
    return filter_signal(data, cutoff=cutoff, sample_rate=sample_rate, filtertype='notch')


def adjust_baseline(lead, fs):
    """
    Shift a lead so its isoelectric line sits at zero. The offset is taken from the flattest
    200 ms window in the recording, which is the best available estimate of the baseline.
    """
    window_size = int(fs * 0.2)
    step_size = int(fs * 0.1)
    least_min_max_diff_sq = np.inf
    adjustment = 0

    for i in np.arange(start=0, stop=len(lead) - window_size, step=step_size):
        window = lead[i:(i + window_size)]
        window_min_max_diff_sq = (np.max(window) - np.min(window)) ** 2

        if window_min_max_diff_sq < least_min_max_diff_sq:
            least_min_max_diff_sq = window_min_max_diff_sq
            adjustment = np.mean(window)

    return lead - adjustment


def filter_lead(lead, filters, fs):
    """Apply the full filter chain to a single lead, in the order given by `filters`."""
    filter_values = derive_filter_values(filters)

    for f in filter_values:
        if f == 'AC':
            lead = filter_signal(lead, filter_values[f], sample_rate=fs, filtertype='notch')
        elif f == 'HP':
            lead = filter_signal(lead, filter_values[f], sample_rate=fs, filtertype='highpass')
        elif f == 'LP':
            lead = filter_signal(lead, filter_values[f], sample_rate=fs, filtertype='lowpass')
        elif f == 'BWR':
            lead = remove_baseline_wander(lead, fs, cutoff=0.05)
        elif f == 'BLA':
            lead = adjust_baseline(lead, fs)
        else:
            raise Exception('Unknown ECG filter: "{}"'.format(f))

    return lead


def load_and_preprocess_ecg(record_id, ecg_filters=None, subsampling_window_size=2000,
                            subsample_start=0, fs=SAMPLING_RATE, src_dir=DEFAULT_ECG_DIR):
    """
    Load one PTB-XL record via wfdb, apply the filter chain, and cut the subsample window.
    Returns a time-major array of shape (subsampling_window_size, num_leads), the model's
    input layout.
    """
    if ecg_filters is None:
        ecg_filters = ECG_FILTERS

    signal, meta = wfdb.rdsamp('{}/{}'.format(src_dir, record_id))
    ecg = perform_shape_switch(np.nan_to_num(signal))  # (num_leads, num_samples)

    for i in range(len(ecg)):
        ecg[i] = filter_lead(ecg[i], ecg_filters, fs)

    ecg = ecg[:, subsample_start:subsample_start + subsampling_window_size]

    return perform_shape_switch(ecg)  # (window, num_leads), time-major


def derive_R_peak_positions(ecg, sampling_rate):
    """Detect R-peak sample positions on the first lead."""
    import neurokit2 as nk

    shp = np.shape(ecg)

    if shp[0] < shp[1]:
        ecg_lead_I = ecg[0]
    else:
        ecg_lead_I = ecg[..., 0]

    return nk.ecg_peaks(np.abs(ecg_lead_I), sampling_rate=sampling_rate, method='neurokit')[1]['ECG_R_Peaks']


def select_r_peak(r_peaks, n_samples, half):
    """Pick the R-peak whose [r-half, r+half] window fits and lies closest to the record centre."""
    valid = [r for r in r_peaks if r - half >= 0 and r + half <= n_samples]

    if not valid:
        raise ValueError('No R-peak leaves room for a +/-{} sample window'.format(half))

    centre = n_samples / 2.0

    return min(valid, key=lambda r: abs(r - centre))


def extract_r_centered_beat(arr_tm, r_peak, half=250):
    """
    Cut an R-centred beat out of a time-major (num_samples, num_leads) array and return it
    lead-major (num_leads, 2*half) with the R-peak at index `half`. The window is taken
    directly around the detected R-peak, so no seam or stitch notch arises.
    """
    window = arr_tm[r_peak - half:r_peak + half, :]

    return perform_shape_switch(window)


def normalize_ecg_relevancemap(R, local=False):
    """Scale a relevance map to [-1, 1], either globally (local=False) or per lead. NaNs become 0."""
    if local is False:
        Rn = R / np.max(np.abs(R))
    else:
        Rn = np.zeros_like(R)
        for i in range(np.shape(R)[1]):
            Rn[..., i] = R[..., i] / np.max(np.abs(R[..., i]))

    return np.nan_to_num(Rn, nan=0)
