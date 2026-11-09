import torch
from torch import nn
from torch.nn import functional as F
import torchaudio


class Frontend(nn.Module):
    """Differentiable waveform-to-log-mel frontend with train-only normalization."""

    def __init__(self, cfg):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(**cfg)
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))

    def raw(self, x):
        """Compute log-mel features and resize them to the classifier input grid."""
        # The complete frontend remains differentiable with respect to waveform samples.
        z = self.mel(x).clamp_min(1e-6).log()
        return F.interpolate(z[:, None], (64, 64), mode="bilinear", align_corners=False)

    def forward(self, x):
        return (self.raw(x) - self.mean) / self.std.clamp_min(1e-6)

    @torch.no_grad()
    def fit(self, waveforms, device):
        """Estimate a single mean/std from training clips only; never use validation/test."""
        total = 0.0
        squared = 0.0
        n = 0
        for x in waveforms.split(32):
            z = self.raw(x.to(device)).double()
            total += z.sum().item()
            squared += z.square().sum().item()
            n += z.numel()
        mean = total / n
        std = max(squared / n - mean * mean, 1e-12) ** 0.5
        self.mean.fill_(mean)
        self.std.fill_(std)


class Recognizer(nn.Module):
    """Waveform classifier; training-only augmentation precedes classification."""

    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        self.frontend = Frontend(cfg["frontend"])
        if cfg["model"] == "tiny_cnn":
            self.classifier = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 16 * 16, num_classes),
            )
        else:
            import timm

            self.classifier = timm.create_model(
                cfg["model"],
                pretrained=False,
                in_chans=1,
                num_classes=num_classes,
                img_size=64,
            )

    def forward(self, x):
        # Augmentation is disabled automatically by model.eval(), including FGSM calls.
        if self.training and self.cfg["time_shift_fraction"] > 0:
            max_shift = int(x.shape[1] * self.cfg["time_shift_fraction"])
            shifts = torch.randint(
                -max_shift, max_shift + 1, (len(x),), device=x.device
            )
            indices = (
                torch.arange(x.shape[1], device=x.device)[None] - shifts[:, None]
            ) % x.shape[1]
            x = x.gather(1, indices)
        z = self.frontend(x)
        if self.training:
            mask = torch.zeros_like(z, dtype=torch.bool)
            for b in range(len(z)):
                for _ in range(self.cfg["masks_per_axis"]):
                    for axis in [2, 3]:
                        width = int(
                            torch.randint(0, self.cfg["mask_max_bins"] + 1, (1,)).item()
                        )
                        if not width:
                            continue
                        start = int(
                            torch.randint(0, z.shape[axis] - width + 1, (1,)).item()
                        )
                        if axis == 2:
                            mask[b, :, start : start + width, :] = True
                        else:
                            mask[b, :, :, start : start + width] = True
            z = torch.where(mask, z.mean((2, 3), keepdim=True), z)
        return self.classifier(z)
