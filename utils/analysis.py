"""
The core global analysis: for every pathology and XAI method it produces the relevance density
figure, the correlation scatter plot, and the two raw metric tables (coverage and correlation)
that all downstream tables are derived from.

Two questions are answered side by side. The correlation analysis asks how strongly a method's
relevance simply tracks signal amplitude, which would make it uninformative. The coverage
analysis asks how much of a method's positive relevance falls inside the segment of the beat
that clinical guidelines consider diagnostic for the pathology at hand.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr as scipy_spearmanr

from utils.beats import calculate_hist_heatmap, load_ecg_beats, load_relevance_beats, normalize_relevance_beats, rotate_to_center
from utils.config import P_SEGMENTS, file_tag, pretty_method_name
from utils.figures import combine_pdfs, plot_population_ecg_row, plot_summary_and_segment_columns


def spearman_correlation(a, b):
    """Spearman rank correlation coefficient, ignoring NaNs."""
    r, p = scipy_spearmanr(a, b, nan_policy='omit')

    return r


def analyze_combined(pathologies, methods, result_dir, plot_dir,
                     posthresh=0.05, cutpos=250, cutoff=1, nleads=12,
                     num_bins=50, seg_len=10, fontsize_dens=45, fontsize_corr=43,
                     wspace=0.05, hspace=0.05, seed=42):
    """
    Run the global analysis over all pathologies and methods.

    Writes one relevance density figure per pathology (hist_<pathology>.pdf) plus their
    concatenation, a single correlation figure spanning all pathologies, and the raw metric
    tables coverage.xlsx and correlation.xlsx. Both metric tables are stored at full precision,
    since every derived table normalises and averages them and must not inherit display rounding.

    Coverage is computed over all data and is therefore exact. The correlation coefficients are
    taken on a random subsample, drawn from a single seeded stream that is consumed in pathology
    order. Running the full set of pathologies reproduces the published coefficients exactly, but
    running a subset shifts which draws each pathology receives, which moves the coefficients by
    a few thousandths. That is well below the two decimals reported, and it leaves coverage, NCov
    and every table derived from them untouched.
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Seed the correlation subsampling, so the reported coefficients are reproducible.
    np.random.seed(seed)

    d_cov = {p: {m: 0 for m in methods} for p in pathologies}
    d_corr = {p: {m: 0 for m in methods} for p in pathologies}

    # squeeze=False keeps the axes array two-dimensional even for a single pathology or method,
    # so the indexing below holds for any subset the caller asks for.
    corr_fig, corr_axs = plt.subplots(len(methods) + 1, len(pathologies),
                                      figsize=(len(pathologies) * 6, (len(methods) + 1) * 3),
                                      squeeze=False)
    corr_fig.subplots_adjust(wspace=wspace, hspace=hspace)

    for i, p in enumerate(pathologies):
        ECG_beats_npy = load_ecg_beats(result_dir, p)
        ECG_beats_abs = np.abs(ECG_beats_npy).ravel()

        dens_fig, dens_axs = plt.subplots(len(methods) + 1, nleads + 2,
                                          figsize=((nleads + 2) * 6, (len(methods) + 1) * 3), sharex=True, squeeze=False)
        dens_fig.subplots_adjust(wspace=wspace, hspace=hspace)

        plot_population_ecg_row(dens_axs[0], ECG_beats_npy, p, cutoff, cutpos, num_bins, seg_len, fontsize_dens)

        for k, m in enumerate(methods):
            print(' ', p, m)
            R_beats = load_relevance_beats(result_dir, p, m)

            # --- Correlation analysis, on the raw absolute values before any clipping ---
            R_beats_abs = np.abs(R_beats).ravel()

            n_sc = min(50000, len(ECG_beats_abs))
            idx_sc = np.random.randint(0, len(ECG_beats_abs), n_sc)
            corr_axs[k + 1][i].scatter(ECG_beats_abs[idx_sc], R_beats_abs[idx_sc], s=2)
            corr_axs[k + 1][i].set_xticks([])
            corr_axs[k + 1][i].set_yticks([])

            if k == len(methods) - 1:
                corr_axs[k + 1][i].set_xlabel(r'$|x_i|$', fontsize=fontsize_corr * 0.5)

            if k == 0:
                corr_axs[k][i].set_title('{} Correlation'.format(p), fontsize=fontsize_corr, pad=17)
                corr_axs[k][i].axis('off')

            if i == 0:
                corr_axs[k + 1][i].set_ylabel(r'$|R_i|$', fontsize=fontsize_corr * 0.5)
                corr_axs[k + 1][i].annotate(pretty_method_name(m), xy=(0, 0.5), xytext=(-corr_axs[k + 1][i].yaxis.labelpad - 10, 0), xycoords=corr_axs[k + 1][i].yaxis.label, textcoords='offset points', ha='right', va='center', rotation=0, fontsize=fontsize_corr)

            # The coefficient is taken on a 500k subsample. Its standard error stays below 0.002,
            # which is far finer than the two decimals reported, at a fraction of the runtime.
            n_scc = min(500000, len(ECG_beats_abs))
            idx_scc = np.random.randint(0, len(ECG_beats_abs), n_scc)
            scc = spearman_correlation(R_beats_abs[idx_scc], ECG_beats_abs[idx_scc])
            d_corr[p][m] = scc
            corr_axs[k + 1][i].text(0.95, 0.95, r'SCC={:.2f}'.format(scc), ha='right', va='top', transform=corr_axs[k + 1][i].transAxes, fontsize=fontsize_corr * 0.5)

            # --- Density and coverage analysis, on the normalised positive relevance ---
            R_beats = normalize_relevance_beats(R_beats, posthresh=posthresh)

            heatmaps = []

            for li in range(nleads):
                R_beats_lead = rotate_to_center(R_beats[:, :, li], cutpos)
                heatmaps.append(calculate_hist_heatmap(R_beats_lead, num_bins, posthresh))

            is_last_row = k == len(methods) - 1

            max_h = np.max(np.array(heatmaps).ravel()) * 0.1

            for li in range(nleads):
                dens_axs[k + 1][li].matshow(heatmaps[li], clim=(0, max_h), aspect='auto')
                dens_axs[k + 1][li].set_xticks([])
                dens_axs[k + 1][li].set_yticks([])

                if is_last_row:
                    dens_axs[k + 1][li].set_xlabel(r'$t$', fontsize=fontsize_dens * 0.7)

                if li == 0:
                    dens_axs[k + 1][li].set_ylabel(r'$R_i^+$', fontsize=fontsize_dens * 0.7)
                    dens_axs[k + 1][li].annotate(pretty_method_name(m), xy=(0, 0.5), xytext=(-dens_axs[k + 1][li].yaxis.labelpad - 15, 0), xycoords=dens_axs[k + 1][li].yaxis.label, textcoords='offset points', ha='right', va='center', rotation=0, fontsize=fontsize_dens)

            R_beats_overall = plot_summary_and_segment_columns(dens_axs[k + 1], R_beats, cutpos, num_bins, seg_len, posthresh, fontsize_dens, is_last_row)

            sum_R = np.sum(R_beats_overall, axis=(0, 1, 2))
            mask_sum = 0

            for s_i, s in enumerate(np.arange(start=0, stop=500, step=seg_len)):
                mask_sum += np.sum(sum_R[s:s + seg_len]) * np.array(P_SEGMENTS[p][s_i])

            d_cov[p][m] = mask_sum / np.sum(np.ravel(R_beats_overall)) * 100

        dens_fig.savefig('{}/hist_{}.pdf'.format(plot_dir, file_tag(p)), orientation='landscape', bbox_inches='tight')
        plt.close(dens_fig)

    corr_fig.savefig('{}/correlation.png'.format(plot_dir), dpi=300, orientation='landscape', bbox_inches='tight')
    plt.close(corr_fig)

    pd.DataFrame(d_cov).to_excel('{}/coverage.xlsx'.format(plot_dir))
    pd.DataFrame(d_corr).to_excel('{}/correlation.xlsx'.format(plot_dir))
    combine_pdfs(['{}/hist_{}.pdf'.format(plot_dir, file_tag(p)) for p in pathologies], '{}/hist_combined.pdf'.format(plot_dir), cleanup=False)
