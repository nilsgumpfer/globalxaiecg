"""
Local explanations for the published example recordings.

One hand-curated PTB-XL record per pathology is explained with the same detector that the
global analysis evaluates, using the pipeline of the preceding AIME2024 study
(https://github.com/nilsgumpfer/AIME2024), so the local figures reproduce that paper's results.
The relevance map is computed over the full recording and only afterwards cut down to a single
R-centred beat, which keeps the explanation identical to the one the model actually produced.
"""
import numpy as np

from utils.config import DEFAULT_ECG_DIR, DEFAULT_MODEL_DIR, EXAMPLE_RECORDS, LAST_CONV_LAYER_NAME, SAMPLING_RATE
from utils.ecg import derive_R_peak_positions, extract_r_centered_beat, load_and_preprocess_ecg, normalize_ecg_relevancemap, select_r_peak
from utils.model import load_detector


def compute_example_beat_and_relevances(pathology, methods, half=250, posthresh=0.2, cmap_adjust=0.3,
                                        last_conv_layer_name=LAST_CONV_LAYER_NAME,
                                        model_dir=DEFAULT_MODEL_DIR, data_dir=DEFAULT_ECG_DIR,
                                        r_peak_index=None):
    """
    For one pathology: load its example record and detector, compute each method's relevance map
    over the full recording, detect R-peaks, and return an R-centred single beat together with
    the matching R-centred, display-ready relevance beats.

    Returns
    -------
    beat_ecg : (num_leads, 2*half) lead-major, R-peak at index `half`
    R_beats  : dict method -> (num_leads, 2*half) display-ready relevance, positives only
    info     : dict with the record id and the chosen R-peak, for logging and captions
    """
    from signxai.methods.wrappers import calculate_relevancemap

    record_id = EXAMPLE_RECORDS[pathology]
    ecg = load_and_preprocess_ecg(record_id, src_dir=data_dir)  # (window, num_leads), time-major
    n_samples = ecg.shape[0]

    model_wo_softmax = load_detector(pathology, model_dir, remove_softmax=True)

    r_peaks = derive_R_peak_positions(ecg, SAMPLING_RATE)

    if r_peak_index is not None:
        r_peak = int(r_peaks[r_peak_index])
    else:
        r_peak = select_r_peak(r_peaks, n_samples, half)

    beat_ecg = extract_r_centered_beat(ecg, r_peak, half=half)

    R_beats = {}

    for method in methods:
        R = np.array(calculate_relevancemap(method, ecg, model_wo_softmax, last_conv_layer_name=last_conv_layer_name))
        R[R < 0] = 0
        Rn = normalize_ecg_relevancemap(R, local=False)
        Rn[Rn <= posthresh] = 0

        # Lift the surviving values away from zero, so that low but relevant evidence still
        # renders as a visible bubble rather than a dot indistinguishable from the baseline.
        Rn[Rn > posthresh] = Rn[Rn > posthresh] + cmap_adjust

        R_beats[method] = extract_r_centered_beat(Rn, r_peak, half=half)

    return beat_ecg, R_beats, {'record_id': record_id, 'r_peak': r_peak, 'r_peaks': list(map(int, r_peaks))}
