"""
All tables and derived statistics reported in the paper, built from the raw coverage.xlsx and
correlation.xlsx written by the global analysis.

Coverage alone is not comparable across pathologies, because the guideline masks differ in width
and a wider mask is easier to hit by chance. Every table therefore reports NCov, coverage
normalised against the random baseline of the same pathology, and the worst case NCov_min across
pathologies, which is what a method has to be judged by if it is to be trusted on any of them.

Values are held at full precision throughout and rounded only where they are rendered, so that no
mean, standard deviation or confidence interval inherits display rounding.
"""
import numpy as np
import pandas as pd

from utils.beats import build_dense_mask, compute_ncov, compute_ncov_exact, compute_pop_cov, load_and_normalize_R_beats, shift_mask
from utils.config import P_SEGMENTS, pretty_method_name


def read_metric_tables(plot_dir, pathologies):
    """Load the raw coverage and correlation tables written by analyze_combined."""
    cov_df = pd.read_excel('{}/coverage.xlsx'.format(plot_dir))
    corr_df = pd.read_excel('{}/correlation.xlsx'.format(plot_dir))
    cov_df.columns = ['Method'] + pathologies
    corr_df.columns = ['Method'] + pathologies

    return cov_df, corr_df


def random_baseline_coverage(cov_df, pathologies):
    """Coverage of the random baseline per pathology, the reference point for NCov."""
    return {p: float(cov_df.loc[cov_df['Method'] == 'random', p].values[0]) for p in pathologies}


def mark_extreme_combined(x, col, max_variant='bf', min_variant='it', decimals=1):
    """
    Format a table column for LaTeX, marking the best value bold and the worst italic. Values
    below ten are padded, so the columns stay optically aligned in the typeset table.
    """
    numeric_vals = x[col][x[col].apply(lambda v: isinstance(v, (int, float)))]

    if numeric_vals.empty:
        return x[col]

    max_val = numeric_vals.max()
    min_val = numeric_vals.min()

    max_formatter = '\\textbf' if max_variant == 'bf' else '\\textit'
    min_formatter = '\\textbf' if min_variant == 'bf' else '\\textit'

    def format_val(val):
        if isinstance(val, (int, float)):
            # Fixed decimal places, so every cell in a column carries the precision the
            # manuscript states.
            s = '{:.{d}f}'.format(val, d=decimals)
            pad = 0 <= val < 10

            if val == max_val:
                return '~~{}{{{}}}'.format(max_formatter, s) if pad else '{}{{{}}}'.format(max_formatter, s)
            elif val == min_val:
                return '~~{}{{{}}}'.format(min_formatter, s) if pad else '{}{{{}}}'.format(min_formatter, s)
            elif pad:
                return '~~{}'.format(s)

            return s

        return '{}'.format(val)

    return x[col].apply(format_val)


def generate_combined_table(plot_dir, pathologies):
    """
    Build the main results table: correlation, raw coverage and normalised coverage side by side
    for every method, sorted by worst-case NCov_min descending.
    """
    cov_df, corr_df = read_metric_tables(plot_dir, pathologies)

    # Keep full-precision sources; cov_df and corr_df get rounded for display only.
    cov_raw = cov_df.copy()
    corr_raw = corr_df.copy()

    cov_random = random_baseline_coverage(cov_raw, pathologies)

    for p in pathologies:
        cov_df[p] = cov_df[p].apply(lambda v: round(float(v), 1))
        corr_df[p] = corr_df[p].apply(lambda v: round(float(v), 2))

    ncov_rows = []
    ncov_exact_rows = []

    for _, row in cov_raw.iterrows():
        ncov_rows.append(dict({'Method': row['Method']}, **{p: compute_ncov(row[p], cov_random[p]) for p in pathologies}))
        ncov_exact_rows.append(dict({'Method': row['Method']}, **{p: compute_ncov_exact(row[p], cov_random[p]) for p in pathologies}))

    ncov_df = pd.DataFrame(ncov_rows)
    ncov_exact_df = pd.DataFrame(ncov_exact_rows)

    # Row means are taken on the unrounded values and rounded only for display, so a mean never
    # inherits the rounding of the cells it is derived from.
    corr_df['Mean'] = corr_raw[pathologies].mean(axis=1).round(2)
    cov_df['Mean'] = cov_raw[pathologies].mean(axis=1).round(1)
    ncov_df['Mean'] = ncov_exact_df[pathologies].mean(axis=1).round(1)

    # Worst-case normalised coverage across pathologies.
    cov_df['Min'] = ncov_exact_df[pathologies].min(axis=1).round(1)

    corr_df['Method'] = corr_df['Method'].apply(pretty_method_name)
    cov_df['Method'] = cov_df['Method'].apply(pretty_method_name)
    ncov_df['Method'] = ncov_df['Method'].apply(pretty_method_name)

    min_ncov_vals = ncov_exact_df[pathologies].min(axis=1)
    sorted_methods = cov_df.assign(_min=min_ncov_vals).sort_values('_min', ascending=False)['Method'].tolist()
    corr_df = corr_df.set_index('Method').loc[sorted_methods].reset_index()
    cov_df = cov_df.set_index('Method').loc[sorted_methods].reset_index()
    ncov_df = ncov_df.set_index('Method').loc[sorted_methods].reset_index()

    for p in pathologies + ['Mean']:
        corr_df[p] = mark_extreme_combined(corr_df, p, decimals=2)
        cov_df[p] = mark_extreme_combined(cov_df, p, decimals=1)
        ncov_df[p] = mark_extreme_combined(ncov_df, p, decimals=1)

    cov_df['Min'] = mark_extreme_combined(cov_df, 'Min', decimals=1)

    n_path = len(pathologies)
    col_layout = 'l' + 'c' * (n_path * 3 + 4)
    path_headers = ' & '.join(['\\textbf{{{}}}'.format(p) for p in pathologies])

    header = '\n'.join([
        r'\begin{tabular}{' + col_layout + '}',
        r'\toprule',
        r'& \multicolumn{' + str(n_path + 1) + r'}{c}{\textbf{Correlation Analysis (SCC)}} '
        r'& \multicolumn{' + str(n_path * 2 + 3) + r'}{c}{\textbf{Pattern Coverage Analysis}} \\',
        r'\cmidrule(lr){2-' + str(n_path + 2) + r'}',
        r'\cmidrule(lr){' + str(n_path + 3) + r'-' + str(n_path * 3 + 5) + r'}',
        r'\textbf{Method}'
        r' & \multicolumn{' + str(n_path) + r'}{c}{\textbf{Pathology}} & \textbf{Mean}'
        r' & \multicolumn{' + str(n_path) + r'}{c}{\textbf{Coverage (TM, \%)}} & \textbf{Mean}'
        r' & \multicolumn{' + str(n_path) + r'}{c}{\textbf{Normalized (NCov, \%)}} & \textbf{Mean}'
        r' & \textbf{$NCov_{min}$} \\',
        r'\cmidrule(lr){2-' + str(n_path + 1) + r'}',
        r'\cmidrule(lr){' + str(n_path + 3) + r'-' + str(n_path * 2 + 2) + r'}',
        r'\cmidrule(lr){' + str(n_path * 2 + 4) + r'-' + str(n_path * 3 + 3) + r'}',
        r'& ' + path_headers + r' & & ' + path_headers + r' & & ' + path_headers + r' & & \\',
        r'\midrule',
    ])

    rows = []

    for idx in range(len(corr_df)):
        method = corr_df.iloc[idx]['Method']
        scc_vals = ' & '.join(str(corr_df.iloc[idx][p]) for p in pathologies)
        scc_mean = str(corr_df.iloc[idx]['Mean'])
        cov_vals = ' & '.join(str(cov_df.iloc[idx][p]) for p in pathologies)
        cov_mean = str(cov_df.iloc[idx]['Mean'])
        ncov_vals = ' & '.join(str(ncov_df.iloc[idx][p]) for p in pathologies)
        ncov_mean = str(ncov_df.iloc[idx]['Mean'])
        min_ncov = str(cov_df.iloc[idx]['Min'])
        rows.append('{} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(method, scc_vals, scc_mean, cov_vals, cov_mean, ncov_vals, ncov_mean, min_ncov))

    latex_table = header + '\n' + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}'

    tex_path = '{}/combined_analysis_table.tex'.format(plot_dir)

    with open(tex_path, 'w') as f:
        f.write(latex_table)

    print('  Saved', tex_path)


def generate_sd_values(plot_dir, pathologies,
                       sign_methods=('lrpsign_epsilon_0_5_std_x', 'smoothgrad_x_sign', 'gradient_x_sign')):
    """
    Cross-pathology dispersion of NCov, reported inline in the manuscript as the SD of NCov.

    ddof=0 is used because the four pathologies are the complete set examined here, not a sample
    drawn from a larger population. NCov is taken unrounded, so the SD does not inherit display
    rounding. The LaTeX macros written alongside are what the manuscript sources pull in.
    """
    cov_df, _ = read_metric_tables(plot_dir, pathologies)
    cov_random = random_baseline_coverage(cov_df, pathologies)

    rows = []

    for _, row in cov_df.iterrows():
        ncov = [compute_ncov_exact(float(row[p]), cov_random[p]) for p in pathologies]
        entry = {'Method': row['Method'], 'SD_NCov': float(np.std(ncov, ddof=0))}
        entry.update({'NCov {}'.format(p): ncov[i] for i, p in enumerate(pathologies)})
        rows.append(entry)

    sd_df = pd.DataFrame(rows)

    xlsx_path = '{}/sd_ncov.xlsx'.format(plot_dir)
    sd_df.to_excel(xlsx_path, index=False)
    print('  Saved', xlsx_path)

    def sd_of(m):
        values = sd_df.loc[sd_df['Method'] == m, 'SD_NCov'].values

        return float(values[0]) if len(values) else None

    # The macros below are what the manuscript sources pull in. They are only emitted for methods
    # actually present, so a run over a reduced method set still produces the table above.
    sign_sds = [v for v in (sd_of(m) for m in sign_methods) if v is not None]
    candidates = [
        (r'\SDGradient', sd_of('gradient')),
        (r'\SDGradientSIGN', sd_of('gradient_x_sign')),
        (r'\SDSignMin', min(sign_sds) if sign_sds else None),
        (r'\SDSignMax', max(sign_sds) if sign_sds else None),
    ]

    macros = [r'% Generated by main.py -- do not edit by hand']
    macros += [r'\newcommand{' + name + '}{' + '{:.1f}'.format(value) + '}' for name, value in candidates if value is not None]

    tex_path = '{}/sd_values.tex'.format(plot_dir)

    with open(tex_path, 'w') as f:
        f.write('\n'.join(macros) + '\n')

    print('  Saved', tex_path)

    for _, r in sd_df.sort_values('SD_NCov').iterrows():
        print('    {:30s} SD(NCov) = {:5.1f}'.format(r['Method'], r['SD_NCov']))


def mask_sensitivity_analysis(pathologies, methods, result_dir, plot_dir,
                              posthresh=0.05, cutpos=250, seg_len=10,
                              offsets=(-20, -10, 0, 10, 20)):
    """
    Recompute NCov_min for every method under symmetric shifts of the guideline mask boundaries
    by k timesteps, to show that the ranking does not hinge on exactly where the masks were drawn.
    The random baseline is recomputed for every shifted mask, since a wider mask raises the
    coverage a random attribution achieves by chance.
    """
    print('Running mask sensitivity analysis...')

    shifted_masks = {p: {k: shift_mask(build_dense_mask(P_SEGMENTS[p], seg_len), k) for k in offsets} for p in pathologies}

    cov_rand = {p: {} for p in pathologies}

    for p in pathologies:
        R_rand = load_and_normalize_R_beats(result_dir, p, 'random', posthresh, cutpos)
        for k in offsets:
            cov_rand[p][k] = compute_pop_cov(R_rand, shifted_masks[p][k])

    ncov_min = {m: {} for m in methods}

    for p in pathologies:
        for m in methods:
            print(' ', p, m)
            R = load_and_normalize_R_beats(result_dir, p, m, posthresh, cutpos)
            for k in offsets:
                cov_mk = compute_pop_cov(R, shifted_masks[p][k])
                cr = cov_rand[p][k]
                ncov_min[m].setdefault(k, {})[p] = (cov_mk - cr) / (100.0 - cr) * 100.0

    for m in methods:
        for k in offsets:
            ncov_min[m][k] = min(ncov_min[m][k].values())

    # Same ordering as the main table, so the two can be read against each other.
    methods_sorted = sorted(methods, key=lambda m: ncov_min[m][0], reverse=True)

    col_labels = [r'$k={:+d}$~ts'.format(k) for k in offsets]
    lines = [
        r'\begin{tabular}{l' + 'r' * len(offsets) + '}',
        r'\toprule',
        r'\textbf{Method} & ' + ' & '.join(col_labels) + r' \\',
        r'\midrule',
    ]

    for m in methods_sorted:
        row = pretty_method_name(m)
        for k in offsets:
            row += r' & {:.1f}'.format(ncov_min[m][k])
        lines.append(row + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}']

    tex_path = '{}/mask_sensitivity_table.tex'.format(plot_dir)

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))

    print('  Saved', tex_path)

    df = pd.DataFrame({'{:+d} ts'.format(k): {m: round(ncov_min[m][k], 1) for m in methods} for k in offsets})
    df.index = [pretty_method_name(m) for m in methods]
    df.to_excel('{}/mask_sensitivity.xlsx'.format(plot_dir))


def bootstrap_confidence_intervals(pathologies, methods, result_dir, plot_dir,
                                   n_bootstrap=1000, posthresh=0.05, cutpos=250,
                                   seg_len=10, ci_level=0.95, seed=42):
    """
    Bootstrap confidence intervals for NCov and NCov_min, resampling at recording level so the
    intervals reflect uncertainty over which patients happened to be in the cohort.

    Coverage is a ratio of sums rather than a mean of per-recording ratios, so the bootstrap
    resamples the numerator and denominator together and recomputes the ratio. This matches the
    metric reported in the main table exactly.
    """
    print('Running bootstrap confidence intervals (n={})...'.format(n_bootstrap))
    np.random.seed(seed)

    lo_pct = (1.0 - ci_level) / 2.0 * 100.0
    hi_pct = 100.0 - lo_pct

    dense_masks = {p: build_dense_mask(P_SEGMENTS[p], seg_len) for p in pathologies}

    # The random baseline is held fixed rather than resampled, so the intervals describe
    # uncertainty in the method under test rather than in the reference point.
    cov_random = {}

    for p in pathologies:
        R_rand = load_and_normalize_R_beats(result_dir, p, 'random', posthresh, cutpos)
        cov_random[p] = compute_pop_cov(R_rand, dense_masks[p])

    cov_num = {p: {} for p in pathologies}
    cov_den = {p: {} for p in pathologies}

    for p in pathologies:
        mask = dense_masks[p]
        for m in methods:
            print(' ', p, m)
            R = load_and_normalize_R_beats(result_dir, p, m, posthresh, cutpos)
            cov_num[p][m] = np.sum(R * mask[np.newaxis, np.newaxis, np.newaxis, :], axis=(1, 2, 3))
            cov_den[p][m] = np.sum(R, axis=(1, 2, 3))

    rows = []

    for m in methods:
        print('  Bootstrapping', m)
        boot_ncov = {p: np.empty(n_bootstrap) for p in pathologies}
        boot_ncov_min = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            ncov_b = []

            for p in pathologies:
                N_p = len(cov_den[p][m])
                idx = np.random.randint(0, N_p, N_p)
                s_num = float(np.sum(cov_num[p][m][idx]))
                s_den = float(np.sum(cov_den[p][m][idx]))
                cov_b = s_num / s_den * 100.0 if s_den > 0 else 0.0
                cr = cov_random[p]
                nv = (cov_b - cr) / (100.0 - cr) * 100.0
                boot_ncov[p][b] = nv
                ncov_b.append(nv)

            boot_ncov_min[b] = min(ncov_b)

        row = {'Method': pretty_method_name(m)}

        for p in pathologies:
            v = boot_ncov[p]
            row['NCov {} mean'.format(p)] = round(float(np.mean(v)), 1)
            row['NCov {} lo'.format(p)] = round(float(np.percentile(v, lo_pct)), 1)
            row['NCov {} hi'.format(p)] = round(float(np.percentile(v, hi_pct)), 1)

        row['NCov_min mean'] = round(float(np.mean(boot_ncov_min)), 1)
        row['NCov_min lo'] = round(float(np.percentile(boot_ncov_min, lo_pct)), 1)
        row['NCov_min hi'] = round(float(np.percentile(boot_ncov_min, hi_pct)), 1)
        rows.append(row)

    pd.DataFrame(rows).set_index('Method').to_excel('{}/bootstrap_ci.xlsx'.format(plot_dir))

    rows_sorted = sorted(rows, key=lambda r: r['NCov_min mean'], reverse=True)
    p_headers = ' & '.join(r'\textbf{{NCov {}}} (95\,\% CI)'.format(p) for p in pathologies)
    lines = [
        r'\begin{tabular}{l' + 'r' * len(pathologies) + 'r}',
        r'\toprule',
        r'\textbf{Method} & ' + p_headers + r' & \textbf{$NCov_{min}$} (95\,\% CI) \\',
        r'\midrule',
    ]

    for row in rows_sorted:
        cells = [row['Method']]

        for p in pathologies:
            cells.append(r'{:.1f} [{:.1f}, {:.1f}]'.format(row['NCov {} mean'.format(p)], row['NCov {} lo'.format(p)], row['NCov {} hi'.format(p)]))

        cells.append(r'{:.1f} [{:.1f}, {:.1f}]'.format(row['NCov_min mean'], row['NCov_min lo'], row['NCov_min hi']))
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}']

    tex_path = '{}/bootstrap_ci_table.tex'.format(plot_dir)

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))

    print('  Saved', tex_path)
