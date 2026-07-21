"""
Numerical primitives operating on the beat arrays that the analysis is built on.

All beat arrays have shape (num_records, num_beats, num_leads, 500). The last axis holds one
resampled heartbeat. The beats are stored R-peak-first, so every consumer rotates them by
`cutpos` timesteps to move the R-peak to the centre of the plotted window before use.
"""
import numpy as np


def load_ecg_beats(result_dir, pathology):
    """Load the raw ECG beat array for one pathology."""
    return np.load('{}/{}/ECG_beats.npy'.format(result_dir, pathology))


def load_relevance_beats(result_dir, pathology, method):
    """Load the raw relevance beat array for one pathology and method, NaNs replaced by zero."""
    R = np.nan_to_num(np.load('{}/{}/R_beats_{}.npy'.format(result_dir, pathology, method)))

    # Collapse an extra class dimension if present, e.g. shape (N, 2, 12, C, T).
    if R.ndim == 5:
        R = R.mean(axis=3)

    return R


def normalize_relevance_beats(R, posthresh=0.05):
    """
    Reduce a relevance beat array to the positive evidence the coverage metric is defined on.
    Negative relevance is clipped away, each recording is scaled to its own maximum so that
    recordings contribute equally regardless of relevance magnitude, and values below
    `posthresh` are dropped as noise. Modifies `R` in place and returns it.
    """
    R[R < 0] = 0

    max_vals = np.abs(R).reshape(len(R), -1).max(axis=1)
    max_vals[max_vals == 0] = 1
    R /= max_vals[:, np.newaxis, np.newaxis, np.newaxis]
    np.nan_to_num(R, copy=False)
    R[R < posthresh] = 0

    return R


def rotate_to_center(arr, cutpos=250):
    """Roll the last axis so the R-peak moves from the array start to the plotted centre."""
    return np.concatenate([arr[..., cutpos:], arr[..., :cutpos]], axis=-1)


def load_and_normalize_R_beats(result_dir, p, m, posthresh=0.05, cutpos=250):
    """Load, normalise and rotate a relevance beat array in one step."""
    R = load_relevance_beats(result_dir, p, m)
    R = normalize_relevance_beats(R, posthresh=posthresh)

    return rotate_to_center(R, cutpos)


def recenter_and_trim(arr, cutpos, cutoff):
    """
    Roll the last axis so the R-peak sits at the seam (cutpos), then trim `cutoff` samples off
    each side of that seam. The beat-extraction window leaves a couple of stray samples at the
    seam that do not line up with the rest of the waveform, producing a visible notch in a single
    real beat's line plot. This is what calculate_mn_beat's `cutoff` already smooths away for the
    population median and quantile band.
    """
    return np.concatenate([arr[..., cutpos:-cutoff], arr[..., cutoff:cutpos]], axis=-1)


def calculate_mn_beat(beats, lead, cutoff, cutpos, norm=True):
    """
    Aggregate a beat array into the median beat and its interquartile band. With lead=None all
    twelve leads are pooled into one summary beat. Returns (q25, median, q75).
    """
    if lead is None:
        leads = np.arange(start=0, stop=12, step=1)
    else:
        leads = [lead]

    values = []

    for l in leads:
        for v in beats:
            for x in v:
                beat_tmp = np.hstack([x[l][cutpos:-cutoff], x[l][cutoff:cutpos]])

                if norm:
                    beat_tmp = beat_tmp / np.max(np.abs(np.ravel(beat_tmp)))

                values.append(beat_tmp)

    return np.nanquantile(values, 0.25, axis=0), np.nanmedian(values, axis=0), np.nanquantile(values, 0.75, axis=0)


def build_dense_mask(seg_list, seg_len=10):
    """Convert a 50-element binary segment list to a dense 500-point float mask."""
    mask = np.zeros(len(seg_list) * seg_len, dtype=float)

    for s_i, val in enumerate(seg_list):
        mask[s_i * seg_len:(s_i + 1) * seg_len] = val

    return mask


def shift_mask(dense_mask, k):
    """
    Expand (k > 0) or contract (k < 0) every contiguous masked region by k timesteps on each
    side. Returns a new float array of the same length.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    bool_mask = dense_mask.astype(bool)

    if k == 0:
        return dense_mask.copy()

    struct = np.ones(abs(k) * 2 + 1, dtype=bool)
    shifted = binary_dilation(bool_mask, structure=struct) if k > 0 else binary_erosion(bool_mask, structure=struct)

    return shifted.astype(float)


def compute_pop_cov(R, dense_mask):
    """
    Population-level coverage, the share of all positive relevance that falls inside the
    guideline-derived mask, in percent.
    """
    num = float(np.sum(R * dense_mask[np.newaxis, np.newaxis, np.newaxis, :]))
    den = float(np.sum(R))

    return num / den * 100.0 if den > 0 else 0.0


def compute_per_sample_cov(R_overall, dense_mask):
    """Per-recording coverage against a dense 500-point mask. Returns an (N,) array in [0, 100]."""
    num = np.sum(R_overall * dense_mask[np.newaxis, np.newaxis, np.newaxis, :], axis=(1, 2, 3))
    den = np.sum(R_overall, axis=(1, 2, 3))

    return np.where(den > 0, num / den * 100.0, 0.0)


def compute_ncov(cov, cov_random):
    """Coverage normalised against the random baseline, rounded for display."""
    return round(compute_ncov_exact(cov, cov_random), 1)


def compute_ncov_exact(cov, cov_random):
    """Unrounded NCov, for derived statistics that must not inherit display rounding."""
    return (cov - cov_random) / (100 - cov_random) * 100


def calculate_hist_heatmap(R_beats, num_bins, posthresh):
    """
    Build the relevance density heatmap of shape (num_bins, T). Each column is the histogram of
    relevance values observed at that timestep across the whole population.
    """
    min_val, max_val = posthresh, 1.0
    T = R_beats.shape[-1]
    flat = R_beats.reshape(-1, T)
    bin_indices = ((flat - min_val) / (max_val - min_val) * num_bins).astype(int)
    valid = (bin_indices >= 0) & (bin_indices < num_bins)
    rows = bin_indices[valid]
    cols = np.broadcast_to(np.arange(T), flat.shape)[valid]
    heatmap = np.bincount(rows * T + cols, minlength=num_bins * T).reshape(num_bins, T)

    return np.flipud(heatmap)


def top_k_pronounced_records(ECG_beats, dense_mask, cutpos, beat_index, k=30):
    """
    Rank recordings by how pronounced their signal is inside the guideline mask, and return the
    top-k (score, record_index) pairs at a fixed beat_index, most pronounced first.
    """
    N = ECG_beats.shape[0]
    mask = dense_mask.astype(bool)
    scored = []

    for i in range(N):
        beat = ECG_beats[i, beat_index]

        if np.isnan(beat).all():
            continue

        beat = np.nan_to_num(beat)
        beat_rc = np.concatenate([beat[:, cutpos:], beat[:, :cutpos]], axis=-1)
        scored.append((np.mean(np.abs(beat_rc[:, mask])), i))

    scored.sort(reverse=True)

    return scored[:k]
