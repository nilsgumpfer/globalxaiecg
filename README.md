# Beyond Local Inspection: Global, Guideline-Grounded Evaluation of Post-hoc XAI Methods for ECG Classification

This repository contains the code and supplementary material accompanying the paper entitled
"Beyond Local Inspection: Global, Guideline-Grounded Evaluation of Post-hoc XAI Methods for ECG
Classification"

Citation:

```bibtex
@Article{Gumpfer2026,
  author  = {Gumpfer, Nils and Guckert, Michael and Sossalla, Samuel and A{\ss}mus, Birgit and Hannig, Jennifer},
  title   = {{Beyond Local Inspection: Global, Guideline-Grounded Evaluation of Post-hoc XAI Methods for ECG Classification}},
  year    = {2026},
}
```

## Summary

Post-hoc XAI methods are usually judged by visual inspection of individual predictions, which
makes the verdict depend on which example happens to be shown, and which cannot reveal whether an
attribution pattern reflects systematic behavior or an instance-specific effect. This work instead
evaluates them **globally**: relevance is aggregated across many heartbeats and compared against
the signal regions that clinical guidelines specify as diagnostic for the condition at hand. The
ECG makes this possible, since those guidelines provide a domain-specific ground truth for what a
trustworthy explanation should emphasize, which is rarely available in other settings.

Two metrics carry the analysis:

- **Coverage (Cov)** — the share of a method's positive relevance that falls inside the
  guideline-derived segment mask.
- **Normalized coverage (NCov)** — the same quantity referenced against a random-attribution
  baseline, so that pathologies with differently sized masks become comparable. Because clinical
  use demands reliability on every condition rather than on average, methods are scored by their
  worst case, **NCov<sub>min</sub>**.

A correlation analysis runs alongside, measuring how strongly a method's relevance merely tracks
signal amplitude, which would make it uninformative regardless of where it lands.

13 gradient-based attribution methods are compared, alongside an Input and a Random baseline, on
four independently trained binary CNNs covering atrioventricular block (AVB), myocardial ischemia
(ISCH), right bundle branch block (RBBB) and left bundle branch block (LBBB). These span the two
categories of ECG diagnostic criteria: low-amplitude intervals and segments near the isoelectric
line (PR interval for AVB, ST segment for ISCH), and high-amplitude QRS morphology patterns
(RBBB, LBBB).

## Setup

The experiments were run with **Python 3.10**. To install the packages and download the data, run:

```bash
./prepare.sh
```

The script installs `requirements.txt`, fetches the four PTB-XL example records used for the local
explanations, and downloads the precomputed beat arrays described below.

## Data

All ECGs come from the PTB-XL database, which is openly accessible via PhysioNet:
https://physionet.org/content/ptb-xl/1.0.3/

The analysis does not run on raw recordings. It runs on **beat arrays**, one file per pathology
and XAI method, of shape `(num_records, num_beats, 12, 500)`. Each entry holds a single heartbeat
resampled to 500 timesteps and aligned to its R-peak, either of the ECG itself (`ECG_beats.npy`)
or of a method's relevance map (`R_beats_<method>.npy`). Two beats are taken per recording, and
only true-positive predictions on the respective test sets enter the analysis.

Producing these arrays means running every XAI method over every true-positive test recording of
every pathology, which takes days of GPU time and draws the cohorts and their splits from the
internal experiment database in which the PTB-XL records were preprocessed and the models were
trained. That pipeline is specific to our infrastructure and is therefore not part of this
repository. The resulting arrays are instead provided as intermediate results and downloaded by
`prepare.sh`, and everything the paper reports is derived from them by the code here.

The archive is a ~2 GB download that unpacks to 12 GB, so make sure the disk has room for both
before running `prepare.sh`. Once unpacked it is laid out as:

```
results/
  AVB/
    ECG_beats.npy
    R_beats_gradient.npy
    R_beats_gradient_x_sign.npy
    ...
  ISCH/
  RBBB/
  LBBB/
```

## Models

The readily trained detectors are provided in `models/`, one per pathology, as a Keras `model.json`
architecture with matching `weights.h5`. These are the four binary CNNs evaluated in the paper,
the same models published with the preceding AIME2024 study
(https://github.com/nilsgumpfer/AIME2024). They are needed only for the local-explanation figures,
which recompute their explanations from scratch.

## Code

Explanations were computed with the `signxai` and `shap` packages:

- SIGN: https://github.com/nilsgumpfer/SIGN-XAI
- SHAP: https://github.com/shap/shap

Only `signxai` is required to run this repository, since it is the one used by the
local-explanation step. The DeepSHAP and GradSHAP attributions reach the analysis through the
precomputed beat arrays, so `shap` itself is not a dependency here.

To reproduce all results, run:

```bash
./start.sh
```

Results are written to `plots/`. Individual parts can be run on their own:

```bash
python3 main.py --steps analysis   # relevance density + correlation figures, raw metrics
python3 main.py --steps tables     # all tables, reusing the raw metrics from the analysis step
python3 main.py --steps local      # local example beat figures
```

Figures are typeset through a local LaTeX installation, as the published ones were. Without LaTeX
available, pass `--no-usetex` to fall back to matplotlib's built-in mathtext. Run
`python3 main.py --help` for the remaining options.

### Repository layout

| Path | Contents |
| --- | --- |
| `main.py` | Entry point and command-line interface |
| `utils/config.py` | Pathologies, methods, guideline segment masks, figure style |
| `utils/beats.py` | Beat array primitives: normalisation, masks, coverage, histograms |
| `utils/analysis.py` | The global analysis producing the density and correlation figures |
| `utils/tables.py` | Main results table, SD values, mask sensitivity, bootstrap CIs |
| `utils/figures.py` | Shared figure rendering and the local-explanation beat figures |
| `utils/ecg.py` | ECG loading, filter chain, R-peak detection, beat extraction |
| `utils/model.py` | Loading the trained detectors |
| `utils/localxai.py` | Local explanations for the published example records |

### Outputs

| File | Content |
| --- | --- |
| `combined_analysis_table.tex` | Main results table: SCC, Cov, NCov and NCov<sub>min</sub> per method |
| `mask_sensitivity_table.tex` | NCov<sub>min</sub> under shifted mask boundaries |
| `bootstrap_ci_table.tex` | Bootstrap confidence intervals for NCov and NCov<sub>min</sub> |
| `sd_values.tex` | Inline SD-of-NCov values referenced by the manuscript |
| `hist_<pathology>.pdf` | Relevance density per method, lead and timestep |
| `correlation.png` | Relevance against signal amplitude, per method and pathology |
| `beat_<pathology>.pdf` | Local explanation of an example beat next to the global pattern |
| `coverage.xlsx`, `correlation.xlsx` | Raw metrics all tables are derived from |

Each `beat_<pathology>.pdf` shows the lead group the condition is diagnosed from, which is the
limb leads I to aVF for AVB and the precordial leads V1 to V6 for the others. This is configured
in `BEAT_FIGURE_LEADS` in `utils/config.py`.

In figure filenames, AVB is spelled out as `AVBLOCK`, so the AVB density figure is
`hist_AVBLOCK.pdf`. The `.tex` tables and `sd_values.tex` are written so the manuscript sources
can `\input` them directly.

Coverage and every table derived from it are exact. The correlation coefficients are taken on a
seeded random subsample, which reproduces the published values exactly for a full run over all
four pathologies. Running a subset shifts them by a few thousandths, well below the two decimals
reported.

## References

### Attribution methods

Adebayo, J., Gilmer, J., Goodfellow, I.J., Kim, B.: Local explanation methods for deep neural networks lack sensitivity to parameter values. In: *6th International Conference on Learning Representations, ICLR 2018, Workshop Track* (2018)

Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.R., Samek, W.: On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. *PLoS One* 10(7), 1–46 (2015)

Erion, G., Janizek, J.D., Sturmfels, P., Lundberg, S.M., Lee, S.: Improving performance of deep learning models with axiomatic attribution priors and expected gradients. *Nature Machine Intelligence* 3, 620–631 (2021)

Gumpfer, N., Prim, J., Keller, T., Seeger, B., Guckert, M., Hannig, J.: SIGNed explanations: Unveiling relevant features by reducing bias. *Information Fusion* 99, 101883 (2023)

Lundberg, S.M., Lee, S.: A unified approach to interpreting model predictions. In: *Advances in Neural Information Processing Systems 30*. pp. 4765–4774 (2017)

Shrikumar, A., Greenside, P., Kundaje, A.: Learning important features through propagating activation differences. In: *Proceedings of the 34th International Conference on Machine Learning, ICML 2017*. Proceedings of Machine Learning Research, vol. 70, pp. 3145–3153. PMLR (2017)

Simonyan, K., Vedaldi, A., Zisserman, A.: Deep inside convolutional networks: Visualising image classification models and saliency maps. In: *2nd International Conference on Learning Representations, ICLR 2014, Workshop Track* (2014)

Smilkov, D., Thorat, N., Kim, B., Viégas, F.B., Wattenberg, M.: SmoothGrad: removing noise by adding noise. *CoRR* abs/1706.03825 (2017)

Sundararajan, M., Taly, A., Yan, Q.: Axiomatic attribution for deep networks. In: *Proceedings of the 34th International Conference on Machine Learning, ICML 2017*. Proceedings of Machine Learning Research, vol. 70, pp. 3319–3328. PMLR (2017)

### Clinical guidelines defining the reference masks

Epstein, A.E., DiMarco, J.P., Ellenbogen, K.A., et al.: ACC/AHA/HRS 2008 guidelines for device-based therapy of cardiac rhythm abnormalities. *Circulation* 117, e350–e408 (2008)

Surawicz, B., Childers, R., Deal, B.J., Gettes, L.S.: AHA/ACCF/HRS recommendations for the standardization and interpretation of the electrocardiogram — part III: Intraventricular conduction disturbances. *Circulation* 119, e235–e240 (2009)

Wagner, G.S., Macfarlane, P., Wellens, H., et al.: AHA/ACCF/HRS recommendations for the standardization and interpretation of the electrocardiogram — part VI: Acute ischemia/infarction. *Circulation* 119, e262–e270 (2009)

### Data, models and tooling

Goldberger, A.L., Amaral, L.A.N., Glass, L., et al.: PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101, e215–e220 (2000)

Gumpfer, N., Grün, D., Hannig, J., Keller, T., Guckert, M.: Detecting myocardial scar using electrocardiogram data and deep neural networks. *Biological Chemistry* 402, 911–923 (2020)

Gumpfer, N., Dinov, B., Sossalla, S., Guckert, M., Hannig, J.: Towards trustworthy AI in cardiology: A comparative analysis of explainable AI methods for electrocardiogram interpretation. In: *22nd International Conference on Artificial Intelligence in Medicine, AIME 2024*. Lecture Notes in Computer Science, vol. 14845, pp. 350–361. Springer (2024)

Makowski, D., Pham, T., Lau, Z.J., et al.: NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods* 53, 1689–1696 (2021)

Wagner, P., Strodthoff, N., Bousseljot, R.D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T.: PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data* 7(1), 154 (2020)
