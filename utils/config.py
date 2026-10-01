"""
Constants and global settings shared across the analysis: the pathologies and XAI methods
evaluated in the paper, the guideline-derived segment masks, the ECG lead order, and the
matplotlib style used for all figures.
"""
import os

import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# Repository root, so all default paths resolve regardless of the caller's working directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DEFAULT_RESULT_DIR = os.path.join(REPO_ROOT, 'results')
DEFAULT_PLOT_DIR = os.path.join(REPO_ROOT, 'plots')
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, 'models')
DEFAULT_ECG_DIR = os.path.join(REPO_ROOT, 'examples', 'ecgs')

# The four pathologies evaluated in the paper.
PATHOLOGIES = ['AVB', 'ISCH', 'RBBB', 'LBBB']

# All XAI methods compared in the main table, including the Input and Random baselines.
METHODS = [
    'gradient',
    'gradient_x_input',
    'gradient_x_sign',
    'smoothgrad',
    'smoothgrad_x_input',
    'smoothgrad_x_sign',
    'vargrad',
    'deep_shap',
    'grad_shap',
    'integrated_gradients',
    'lrp_alpha_1_beta_0',
    'lrp_epsilon_0_5_std_x',
    'lrpsign_epsilon_0_5_std_x',
    'input',
    'random',
]

# Methods shown in the local-explanation beat figures.
LOCAL_METHODS = ['lrp_epsilon_0_5_std_x', 'lrpsign_epsilon_0_5_std_x']

# Standard 12-lead order of the beat arrays.
LEAD_INDEX = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Lead groups available to the local-explanation beat figures, as (array index, label) pairs.
LIMB_LEADS = [(0, 'I'), (1, 'II'), (2, 'III'), (3, 'aVL'), (4, 'aVR'), (5, 'aVF')]
PRECORDIAL_LEADS = [(6, 'V1'), (7, 'V2'), (8, 'V3'), (9, 'V4'), (10, 'V5'), (11, 'V6')]

# Which lead group each pathology's beat figure shows. AVB is diagnosed from PR-interval
# prolongation, which is assessed in the limb leads, so its figure shows I to aVF. The remaining
# pathologies are read from the precordial leads, which is the default.
BEAT_FIGURE_LEADS = {
    'AVB': LIMB_LEADS,
}


def beat_figure_leads(pathology):
    """Lead group shown in the local-explanation beat figure for a pathology."""
    return BEAT_FIGURE_LEADS.get(pathology, PRECORDIAL_LEADS)

# Guideline-derived temporal segment masks. Each list holds 50 bins of 10 timesteps
# (500 timesteps per resampled beat, in the rotated frame with the R-peak at the seam).
# A 1 marks a timestep that the clinical guidelines consider diagnostic for the pathology.
P_SEGMENTS = {
    'AVB':  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ISCH': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'RBBB': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LBBB': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# PTB-XL record used as the local-explanation example per pathology (500 Hz, "_hr"). These are
# the records published with AIME2024 (https://github.com/nilsgumpfer/AIME2024), explained with
# the same models as the detectors evaluated here.
EXAMPLE_RECORDS = {
    'AVB': '03509_hr',
    'ISCH': '12131_hr',
    'LBBB': '14493_hr',
    'RBBB': '02906_hr',
}

# Filter chain applied to every raw PTB-XL recording before it reaches the model.
ECG_FILTERS = ['BWR', 'BLA', 'AC50Hz', 'LP40Hz']

SAMPLING_RATE = 500

# Name of the last convolutional layer, needed by the Grad-CAM style methods in signxai.
LAST_CONV_LAYER_NAME = 'activation_4'

# True while matplotlib renders text through a real LaTeX installation. Set by set_plot_style().
USETEX = True


def file_tag(pathology):
    """
    Filename tag for a pathology. AVB is spelled out, since the figure filenames referenced by
    the manuscript sources use the long form.
    """
    return pathology.replace('AVB', 'AVBLOCK')


def set_plot_style(usetex=True, fontsize=14):
    """
    Apply the figure style used for the paper. With usetex=True matplotlib renders all text
    through a local LaTeX installation, which is what produced the published figures. Set
    usetex=False to fall back to matplotlib's built-in mathtext if no LaTeX is available; the
    figures then still render, but symbols and bold face may deviate slightly.
    """
    global USETEX
    USETEX = usetex

    rcParams['text.usetex'] = usetex
    rcParams['font.size'] = fontsize
    rcParams['savefig.format'] = 'pdf'

    if usetex:
        rcParams['text.latex.preamble'] = '\\usepackage{amssymb}\n \\usepackage{amsmath}'


def bold(text):
    """Bold a piece of plain text in whichever text renderer is active."""
    if USETEX:
        return r'\textbf{' + text + '}'

    return r'$\bf{' + text.replace(' ', r'\ ') + '}$'


def register_fair_reds_cmap():
    """Register the white-to-red colormap used for the relevance bubbles."""
    try:
        mpl.colormaps.register(LinearSegmentedColormap.from_list('FairReds', [(1, 1, 1), (1, 0, 0)], N=256), name='FairReds')
    except ValueError:
        pass


def pretty_method_name(m):
    """Map an internal method key to the LaTeX label used in the paper's figures and tables."""
    mapping = {
        'random': r'Random',
        'input': r'Input',
        'gradient': r'Gradient',
        'grad_shap': r'GradSHAP',
        'deep_shap': r'DeepSHAP',
        'gradient_x_input': r'Gradient $\times$ Input',
        'gradient_x_sign': r'Gradient $\times$ SIGN',
        'smoothgrad_x_input': r'SmoothGrad $\times$ Input',
        'smoothgrad_x_sign': r'SmoothGrad $\times$ SIGN',
        'deconvnet_x_sign': r'DeconvNet $\times$ SIGN',
        'guided_backprop_x_sign': r'Guided Backpropagation $\times$ SIGN',
        'integrated_gradients': r'Integrated Gradients',
        'smoothgrad': r'SmoothGrad',
        'vargrad': r'VarGrad',
        'deconvnet': r'DeconvNet',
        'guided_backprop': r'Guided Backpropagation',
        'grad_cam_timeseries': r'Grad-CAM',
        'guided_grad_cam_timeseries': r'Grad-CAM',
        'lrp_epsilon_0_1_std_x': r'LRP-$\epsilon~(\epsilon = 0.1 \cdot \sigma(x))$',
        'lrp_epsilon_0_5_std_x': r'LRP-$\epsilon~(\epsilon = 0.5 \cdot \sigma(x))$',
        'lrpsign_epsilon_0_1_std_x': r'LRP-$\epsilon~(\epsilon = 0.1 \cdot \sigma(x))$/SIGN',
        'lrpsign_epsilon_0_5_std_x': r'LRP-$\epsilon~(\epsilon = 0.5 \cdot \sigma(x))$/SIGN',
        'lrp_alpha_1_beta_0': r'LRP-$\alpha\beta$',
    }

    try:
        return mapping[m]
    except KeyError:
        return m.replace('_', r'\_')
