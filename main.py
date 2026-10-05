"""
Reproduce the results of the paper on global, guideline-grounded evaluation of XAI methods for
ECG interpretation.

The analysis runs on precomputed beat-wise arrays, one per pathology and XAI method, which are
downloaded separately because of their size (see prepare.sh). Everything the paper reports is
derived from them here: the relevance density figures, the correlation figures, the main results
table, the mask sensitivity and bootstrap supplementary tables, and the inline SD values.

The local example figures are the one step that does not read the precomputed arrays. They load
a published PTB-XL record and its detector and compute the explanation from scratch, so they need
TensorFlow and signxai. Every other step needs only numpy, pandas, matplotlib and scipy.

Usage
-----
    python main.py                       # everything the paper reports
    python main.py --steps analysis      # only the density and correlation figures + raw metrics
    python main.py --steps tables        # only the tables, reusing existing raw metrics
    python main.py --steps local         # only the local example beat figures
    python main.py --no-usetex           # render without a LaTeX installation
"""
import argparse
import os

from utils.config import DEFAULT_MODEL_DIR, DEFAULT_PLOT_DIR, DEFAULT_RESULT_DIR, LOCAL_METHODS, METHODS, PATHOLOGIES, set_plot_style

ALL_STEPS = ['analysis', 'tables', 'local']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', nargs='+', default=ALL_STEPS, choices=ALL_STEPS + ['all'],
                        help='Which parts of the analysis to run (default: all).')
    parser.add_argument('--result-dir', default=DEFAULT_RESULT_DIR,
                        help='Directory holding the downloaded beat arrays, one subdirectory per pathology.')
    parser.add_argument('--plot-dir', default=DEFAULT_PLOT_DIR,
                        help='Directory the figures and tables are written to.')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                        help='Directory holding the trained detectors, one subdirectory per pathology.')
    parser.add_argument('--pathologies', nargs='+', default=PATHOLOGIES,
                        help='Pathologies to evaluate.')
    parser.add_argument('--methods', nargs='+', default=METHODS,
                        help='XAI methods to evaluate.')
    parser.add_argument('--local-methods', nargs='+', default=LOCAL_METHODS,
                        help='XAI methods shown in the local example beat figures.')
    parser.add_argument('--no-usetex', dest='usetex', action='store_false',
                        help='Render text with matplotlib mathtext instead of a LaTeX installation.')
    parser.add_argument('--gpu', default=None,
                        help='GPU id to make visible to TensorFlow. Omit to leave the environment untouched, '
                             'or pass -1 to force CPU. Only affects the local example figures.')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap resamples for the confidence intervals.')

    return parser.parse_args()


def check_result_dir(result_dir, pathologies):
    """Fail early with an actionable message if the downloaded beat arrays are not in place."""
    missing = [p for p in pathologies if not os.path.isfile('{}/{}/ECG_beats.npy'.format(result_dir, p))]

    if missing:
        raise SystemExit(
            'No beat arrays found for {} in "{}".\n'
            'These intermediate results are downloaded separately because of their size. '
            'Run ./prepare.sh, or pass --result-dir to point at an existing copy.'.format(', '.join(missing), result_dir))


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    steps = ALL_STEPS if 'all' in args.steps else args.steps

    set_plot_style(usetex=args.usetex)
    os.makedirs(args.plot_dir, exist_ok=True)
    check_result_dir(args.result_dir, args.pathologies)

    if 'analysis' in steps:
        from utils.analysis import analyze_combined

        print('Running global analysis...')
        analyze_combined(args.pathologies, args.methods, args.result_dir, args.plot_dir)

    if 'tables' in steps:
        from utils.tables import bootstrap_confidence_intervals, generate_combined_table, generate_sd_values, mask_sensitivity_analysis, scc_confidence_intervals

        print('Generating tables...')
        generate_combined_table(args.plot_dir, args.pathologies)
        generate_sd_values(args.plot_dir, args.pathologies)
        mask_sensitivity_analysis(args.pathologies, args.methods, args.result_dir, args.plot_dir)
        bootstrap_confidence_intervals(args.pathologies, args.methods, args.result_dir, args.plot_dir, n_bootstrap=args.n_bootstrap)
        scc_confidence_intervals(args.pathologies, args.methods, args.result_dir, args.plot_dir, n_bootstrap=args.n_bootstrap)

    if 'local' in steps:
        from utils.figures import plot_example_beat_figures

        print('Rendering local example beat figures...')
        plot_example_beat_figures(args.pathologies, args.local_methods, args.result_dir, args.plot_dir, model_dir=args.model_dir)

    print('Done. Results written to {}'.format(args.plot_dir))


if __name__ == '__main__':
    main()
