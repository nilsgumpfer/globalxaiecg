"""
Figure rendering shared by the global analysis and the local example figures.

Two row types recur throughout the paper's figures. The population row shows the median beat
with its interquartile band per lead, a pooled summary beat, and the guideline segment mask.
The per-method row shows either the relevance density across the population or, in the local
figures, a single beat with its relevance drawn as bubbles on the waveform.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter

from utils.beats import calculate_hist_heatmap, calculate_mn_beat, load_ecg_beats, load_relevance_beats, normalize_relevance_beats, recenter_and_trim, rotate_to_center, top_k_pronounced_records, build_dense_mask
from utils.config import LEAD_INDEX, P_SEGMENTS, beat_figure_leads, bold, file_tag, pretty_method_name, register_fair_reds_cmap


def combine_pdfs(paths, targetpath, cleanup=False):
    """Concatenate several PDF files into one."""
    pdf_writer = PdfWriter()

    for path in paths:
        pdf_reader = PdfReader(path)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    with open(targetpath, 'wb') as fh:
        pdf_writer.write(fh)

    if cleanup:
        for path in paths:
            os.remove(path)


def plot_population_ecg_row(axs, ECG_beats_npy, pathology, cutoff, cutpos, num_bins, seg_len, fontsize, lead_labels=None):
    """
    Draw the top row of a figure: the median beat with its interquartile band per lead, a pooled
    summary beat, and the guideline segment mask with the median beat overlaid in white.
    Returns the pooled median beat and its maximum, which callers reuse for scaling.
    """
    if lead_labels is None:
        lead_labels = [(li, LEAD_INDEX[li]) for li in range(len(LEAD_INDEX))]

    for i, (li, ln) in enumerate(lead_labels):
        q025_li, med_li, q075_li = calculate_mn_beat(ECG_beats_npy, lead=li, norm=True, cutoff=cutoff, cutpos=cutpos)
        med_li = med_li / np.max(np.abs(med_li))
        q025_li = q025_li / np.max(np.abs(q025_li))
        q075_li = q075_li / np.max(np.abs(q075_li))
        axs[i].plot(q025_li, linewidth=1, color='silver')
        axs[i].plot(q075_li, linewidth=1, color='silver')
        axs[i].fill_between(np.arange(len(q025_li)), q025_li, q075_li, color='silver')
        axs[i].plot(med_li, linewidth=3, color='k')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_ylim((-1.1, 1.1))
        axs[i].set_title('Lead {}'.format(ln), fontsize=fontsize, rotation=0, horizontalalignment='center', verticalalignment='bottom')

    q025_all, med_all, q075_all = calculate_mn_beat(ECG_beats_npy, lead=None, norm=True, cutoff=cutoff, cutpos=cutpos)
    median_max = np.max(np.abs(med_all))
    axs[-2].plot(q025_all, linewidth=1, color='silver')
    axs[-2].plot(q075_all, linewidth=1, color='silver')
    axs[-2].fill_between(np.arange(len(q025_all)), q025_all, q075_all, color='silver')
    axs[-2].plot(med_all, linewidth=3, color='k')
    axs[-2].set_xticks([])
    axs[-2].set_yticks([])
    axs[-2].set_ylim((-median_max * 1.1, median_max * 1.1))
    axs[-2].set_title('Summary', fontsize=fontsize, rotation=0, horizontalalignment='center', verticalalignment='bottom')

    segment_ECG = np.zeros((num_bins, 500))

    for s_i, s in enumerate(np.arange(start=0, stop=500, step=seg_len)):
        segment_ECG[..., s:s + seg_len] = P_SEGMENTS[pathology][s_i]

    axs[-1].matshow(segment_ECG, aspect='auto')
    axs[-1].plot(-med_all / median_max * 0.9 * num_bins / 2 + num_bins / 2, color='white', linewidth=3)
    axs[-1].set_xticks([])
    axs[-1].set_yticks([])
    axs[-1].set_title('{} Segments'.format(pathology), fontsize=fontsize, rotation=0, horizontalalignment='center', verticalalignment='bottom')

    return med_all, median_max


def plot_summary_and_segment_columns(axs, R_beats, cutpos, num_bins, seg_len, posthresh, fontsize, is_last_row):
    """
    Draw the two rightmost columns of a method row: the relevance density pooled over all leads,
    and the same relevance averaged within each guideline segment. Returns the rotated relevance
    array, on which the coverage metric is computed.
    """
    R_beats_overall = rotate_to_center(R_beats, cutpos)
    mean_heatmap = calculate_hist_heatmap(R_beats_overall, num_bins, posthresh)
    max_h_mean = np.max(np.array(mean_heatmap).ravel()) * 0.1
    axs[-2].matshow(mean_heatmap, clim=(0, max_h_mean), aspect='auto')
    axs[-2].set_xticks([])
    axs[-2].set_yticks([])

    if is_last_row:
        axs[-2].set_xlabel(r'$t$', fontsize=fontsize * 0.7)

    mean_R = np.mean(R_beats_overall, axis=(0, 1, 2))
    segment_heatmap = np.zeros(mean_heatmap.shape, dtype=float)

    for s in np.arange(start=0, stop=500, step=seg_len):
        segment_heatmap[..., s:s + seg_len] = np.mean(mean_R[s:s + seg_len])

    axs[-1].matshow(segment_heatmap, aspect='auto', clim=(0, np.quantile(segment_heatmap.ravel(), 0.99)), cmap='viridis')
    axs[-1].set_xticks([])
    axs[-1].set_yticks([])

    if is_last_row:
        axs[-1].set_xlabel(r'$t$', fontsize=fontsize * 0.7)

    return R_beats_overall


def plot_bubble_beat(ax, ecg_lead, relevance_lead, bubble_size=100, clim_max=1.0, linewidth=2.5):
    """
    Plot a single ECG lead beat with its positive relevance overlaid as bubbles. Bubble size and
    colour both encode relevance, and bubbles are drawn in ascending order, so the strongest
    evidence ends up on top rather than hidden behind weaker neighbours.
    """
    x = np.arange(len(ecg_lead))
    ax.plot(x, ecg_lead, linewidth=linewidth, color='k', zorder=3)

    mask = relevance_lead > 0

    if np.any(mask):
        order = np.argsort(relevance_lead[mask])
        x_dots = x[mask][order]
        y_dots = ecg_lead[mask][order]
        z_dots = relevance_lead[mask][order]
        sizes = (np.abs(z_dots) / clim_max) * bubble_size
        ax.scatter(x_dots, y_dots, marker='o', c=z_dots, cmap='FairReds', s=sizes, zorder=2, vmin=0, vmax=clim_max)


def render_single_beat_figure(p, rec_idx, sel_beat_idx, methods, result_dir, save_path,
                              posthresh=0.05, cmap_adjust=0.3, cutpos=250, cutoff=1, local_cutoff=3,
                              num_bins=50, seg_len=10, fontsize_dens=45,
                              wspace=0.05, hspace=0.05, bubble_size=200, beat_linewidth=2.5,
                              beat_ecg_override=None, R_beat_override=None, leads=None):
    """
    Render one local-explanation beat figure. The lead columns show a single beat with its
    relevance as bubbles, while the two rightmost columns keep showing the population-level
    density, so a local explanation can be read directly against the global pattern.

    `leads` selects which leads are shown, as (array index, label) pairs. It defaults to the group
    the pathology is diagnosed from, which is the limb leads for AVB and the precordial leads
    otherwise.

    `beat_ecg_override` and `R_beat_override` let a caller inject the single-beat cells from a
    different source than the population beats in `result_dir`. This is how the published example
    records are shown: an R-centred beat taken straight from the recording, explained with the
    full pipeline, while the top row and the Summary and Segments columns still come from the
    population data. An injected relevance beat is expected to be display-ready already, so no
    population normalisation is applied to it.
    """
    register_fair_reds_cmap()

    if leads is None:
        leads = beat_figure_leads(p)

    nleads = len(leads)

    ECG_beats_npy = load_ecg_beats(result_dir, p)

    if beat_ecg_override is not None:
        beat_ecg = np.nan_to_num(beat_ecg_override)  # (num_leads, T), already R-centred
    else:
        beat_ecg = np.nan_to_num(ECG_beats_npy[rec_idx, sel_beat_idx])
        beat_ecg = recenter_and_trim(beat_ecg, cutpos, local_cutoff)

    beat_ecg_norm = np.zeros_like(beat_ecg)

    for li, ln in leads:
        denom = np.max(np.abs(beat_ecg[li]))
        beat_ecg_norm[li] = beat_ecg[li] / denom if denom > 0 else beat_ecg[li]

    dens_fig, dens_axs = plt.subplots(len(methods) + 1, nleads + 2, figsize=((nleads + 2) * 6, (len(methods) + 1) * 3), sharex=True, squeeze=False)
    dens_fig.subplots_adjust(wspace=wspace, hspace=hspace)

    plot_population_ecg_row(dens_axs[0], ECG_beats_npy, p, cutoff, cutpos, num_bins, seg_len, fontsize_dens, lead_labels=leads)

    # Left-margin row label for the overview row, matching the method-row labels below.
    dens_axs[0][0].set_ylabel(' ', fontsize=fontsize_dens * 0.7)
    dens_axs[0][0].annotate('Median Beat', xy=(0, 0.5), xytext=(-dens_axs[0][0].yaxis.labelpad - 15, 0), xycoords=dens_axs[0][0].yaxis.label, textcoords='offset points', ha='right', va='center', rotation=0, fontsize=fontsize_dens)

    for k, m in enumerate(methods):
        print('  beat plot', p, m)
        R_beats = load_relevance_beats(result_dir, p, m)

        # Same normalisation as the density analysis, so the Summary and Segments columns match
        # hist_<pathology>.pdf exactly.
        R_beats = normalize_relevance_beats(R_beats, posthresh=posthresh)

        R_beat = recenter_and_trim(R_beats[rec_idx, sel_beat_idx].copy(), cutpos, local_cutoff)

        # Lift positives away from zero for display only, so weak evidence stays visible.
        R_beat_display = R_beat.copy()
        amp_mask = R_beat_display > 0
        R_beat_display[amp_mask] = R_beat_display[amp_mask] + cmap_adjust

        if R_beat_override is not None:
            R_beat_display = np.nan_to_num(R_beat_override[m])

        is_last_row = k == len(methods) - 1

        for i, (li, ln) in enumerate(leads):
            plot_bubble_beat(dens_axs[k + 1][i], beat_ecg_norm[li], R_beat_display[li], bubble_size=bubble_size, clim_max=1.0, linewidth=beat_linewidth)
            dens_axs[k + 1][i].set_xticks([])
            dens_axs[k + 1][i].set_yticks([])
            dens_axs[k + 1][i].set_ylim((-1.1, 1.1))

            if is_last_row:
                dens_axs[k + 1][i].set_xlabel(r'$t$', fontsize=fontsize_dens * 0.7)

            if i == 0:
                dens_axs[k + 1][i].set_ylabel(r'$x_i$', fontsize=fontsize_dens * 0.7)
                dens_axs[k + 1][i].annotate(pretty_method_name(m), xy=(0, 0.5), xytext=(-dens_axs[k + 1][i].yaxis.labelpad - 15, 0), xycoords=dens_axs[k + 1][i].yaxis.label, textcoords='offset points', ha='right', va='center', rotation=0, fontsize=fontsize_dens)

        plot_summary_and_segment_columns(dens_axs[k + 1], R_beats, cutpos, num_bins, seg_len, posthresh, fontsize_dens, is_last_row)

    _draw_section_headers(dens_fig, dens_axs, nleads, fontsize_dens)

    dens_fig.savefig(save_path, orientation='landscape', bbox_inches='tight')
    plt.close(dens_fig)


def _draw_section_headers(dens_fig, dens_axs, nleads, fontsize):
    """
    Label the two column groups above the figure. The lead columns hold the per-beat local
    explanations, the last two columns the population-level attribution patterns.
    """
    try:
        dens_fig.canvas.draw()
        renderer = dens_fig.canvas.get_renderer()
        inv = dens_fig.transFigure.inverted()

        def _title_top(ax):
            return inv.transform(ax.title.get_window_extent(renderer))[1][1]
    except Exception:
        # No renderer available, so fall back to estimating the title height from the font size.
        _title_h = fontsize / 72.0 / dens_fig.get_figheight()

        def _title_top(ax):
            return ax.get_position().y1 + _title_h

    def _section_header(col_axes, label):
        x0 = min(ax.get_position().x0 for ax in col_axes)
        x1 = max(ax.get_position().x1 for ax in col_axes)
        yc = max(_title_top(ax) for ax in col_axes) + 0.02
        dens_fig.text((x0 + x1) / 2.0, yc, label, ha='center', va='bottom', fontsize=fontsize)

    _section_header([dens_axs[0][i] for i in range(nleads)], bold('Local Explanations'))
    _section_header([dens_axs[0][-2], dens_axs[0][-1]], bold('Global Attribution Patterns'))


def plot_example_beat_figures(pathologies, methods, result_dir, plot_dir, half=250, model_dir=None, **kwargs):
    """
    Render the local-explanation beat figures whose single-beat cells come from the published
    example records, one per pathology. The beat is R-centred and the explanation is computed
    over the full recording, so no beat-extraction seam enters the local view. The top row and
    the Summary and Segments columns still come from the population data for comparison.
    """
    from utils.config import DEFAULT_MODEL_DIR
    from utils.localxai import compute_example_beat_and_relevances

    os.makedirs(plot_dir, exist_ok=True)

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    for p in pathologies:
        beat_ecg, R_beats, info = compute_example_beat_and_relevances(p, methods, half=half, model_dir=model_dir)
        print('  example beat', p, info['record_id'], 'r_peak', info['r_peak'], 'of', info['r_peaks'])

        save_path = '{}/beat_{}.pdf'.format(plot_dir, file_tag(p))

        # rec_idx and sel_beat_idx are unused when overrides are supplied, but the population
        # arrays behind the top row and the two rightmost columns are still read from result_dir.
        render_single_beat_figure(p, 0, 0, methods, result_dir, save_path,
                                  beat_ecg_override=beat_ecg, R_beat_override=R_beats, **kwargs)


def plot_single_beat_candidates(p, methods, result_dir, plot_dir, beat_index=1, k=30, seg_len=10, cutpos=250, **kwargs):
    """
    Render one beat figure per candidate for the top-k recordings whose signal is most pronounced
    inside the guideline mask, so an illustrative beat can be picked by visual review rather than
    by a heuristic alone.
    """
    os.makedirs(plot_dir, exist_ok=True)

    ECG_beats_npy = load_ecg_beats(result_dir, p)
    dense_mask = build_dense_mask(P_SEGMENTS[p], seg_len)
    top = top_k_pronounced_records(ECG_beats_npy, dense_mask, cutpos, beat_index, k=k)

    for rank, (score, rec_idx) in enumerate(top, start=1):
        print('  candidate {}/{}: idx={} score={:.3f}'.format(rank, len(top), rec_idx, score))
        save_path = '{}/beat_{}_idx{}_b{}.pdf'.format(plot_dir, file_tag(p), rec_idx, beat_index)
        render_single_beat_figure(p, rec_idx, beat_index, methods, result_dir, save_path, cutpos=cutpos, seg_len=seg_len, **kwargs)
