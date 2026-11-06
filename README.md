
**Evaluating Acoustic Keystroke Recognition under Random and Adversarial Noise**

This project investigates how added noise affects the performance of a deep learning model that identifies keyboard keys from their sounds. It is inspired by *A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards* by Harrison et al. (2023), which demonstrated the feasibility of acoustic keystroke recognition using recordings from ordinary devices.

Our research question is: *At the same signal-to-noise ratio (SNR), does gradient-based adversarial noise reduce keystroke recognition accuracy more than random Gaussian noise?* We hypothesize that SNR-calibrated FGSM-sign perturbations will cause a greater reduction, with **20 dB** as the primary comparison.

We will use the publicly available "Phone" dataset (https://github.com/JBFH-Dev/Keystroke-Datasets.git), covering 36 alphanumeric keys. Recordings will be segmented into individual keystrokes and manually checked before a stratified 60/20/20 training, validation, and test split. A CoAtNet-based classifier implemented with PyTorch/timm will be trained on log-Mel spectrograms and selected using validation performance.

The selected model will remain fixed while the same test samples are evaluated under clean conditions, Gaussian noise, and FGSM-sign perturbations at **30, 20, and 10 dB**. Noise levels will be matched using measured SNR after clipping. Evaluation will include accuracy, macro-F1, accuracy reduction, and attack success rate among initially correct predictions. Three Gaussian seeds and paired bootstrap confidence intervals will support the comparison.

The expected contribution is a reproducible assessment of the model’s sensitivity to random and adversarial perturbations. Conclusions will be limited to a digital, white-box experiment within the recorded environment.
