# VAL results

Exploratory validation diagnostics; validation was also used for model selection.

| Condition | SNR (dB) | Accuracy (%) | Macro-F1 | Drop (pp) | ASR (%) |
|---|---:|---:|---:|---:|---:|
| Clean | -- | 96.11 | 0.9608 | 0.00 | -- |
| Gaussian | 60 | 93.52 ± 0.32 | 0.9306 | 2.59 | 3.85 |
| Gaussian | 50 | 72.41 ± 1.40 | 0.7054 | 23.70 | 26.01 |
| Gaussian | 40 | 24.44 ± 1.47 | 0.1741 | 71.67 | 76.30 |
| Gaussian | 35 | 9.63 ± 0.32 | 0.0438 | 86.48 | 91.71 |
| Gaussian | 30 | 5.19 ± 0.85 | 0.0130 | 90.93 | 95.76 |
| Gaussian | 20 | 2.78 ± 0.00 | 0.0016 | 93.33 | 97.11 |
| Gaussian | 10 | 3.89 ± 1.47 | 0.0045 | 92.22 | 96.34 |
| FGSM-sign | 60 | 16.67 | 0.1586 | 79.44 | 82.66 |
| FGSM-sign | 50 | 20.00 | 0.1875 | 76.11 | 79.19 |
| FGSM-sign | 40 | 9.44 | 0.0477 | 86.67 | 91.33 |
| FGSM-sign | 35 | 5.00 | 0.0076 | 91.11 | 95.95 |
| FGSM-sign | 30 | 4.44 | 0.0081 | 91.67 | 95.95 |
| FGSM-sign | 20 | 3.33 | 0.0066 | 92.78 | 96.53 |
| FGSM-sign | 10 | 3.33 | 0.0031 | 92.78 | 96.53 |

Gaussian ± denotes sample standard deviation across three streams derived from master seed 26, not a confidence interval. ASR is conditional on clean-correct clips.

Paired effect: Gaussian accuracy minus FGSM accuracy. See paired_effects.csv for stratified paired bootstrap intervals.
