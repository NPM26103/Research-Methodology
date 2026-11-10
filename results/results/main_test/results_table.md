# TEST results

Main test experiment; the primary comparison is at 20 dB.

| Condition | SNR (dB) | Accuracy (%) | Macro-F1 | Drop (pp) | ASR (%) |
|---|---:|---:|---:|---:|---:|
| Clean | -- | 93.89 | 0.9370 | 0.00 | -- |
| Gaussian | 30 | 4.07 ± 1.28 | 0.0078 | 89.81 | 95.66 |
| Gaussian | 20 | 2.78 ± 0.00 | 0.0016 | 91.11 | 97.04 |
| Gaussian | 10 | 3.15 ± 0.32 | 0.0035 | 90.74 | 96.65 |
| FGSM-sign | 30 | 4.44 | 0.0092 | 89.44 | 95.27 |
| FGSM-sign | 20 | 2.78 | 0.0016 | 91.11 | 97.04 |
| FGSM-sign | 10 | 4.44 | 0.0047 | 89.44 | 95.86 |

Gaussian ± denotes sample standard deviation across three streams derived from master seed 26, not a confidence interval. ASR is conditional on clean-correct clips.

Paired effect: Gaussian accuracy minus FGSM accuracy. See paired_effects.csv for stratified paired bootstrap intervals.
