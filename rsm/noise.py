"""Waveform perturbations for an offline, label-aware robustness study.

FGSM-sign directions are calibrated to measured SNR after clipping. This
is not a fixed-epsilon Linf benchmark or an optimal L2 attack.
"""

import math
import torch
import torch.nn.functional as F


def _check(x):
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("Expected floating tensor [B,T]")
    if not torch.isfinite(x).all() or x.abs().max() > 1:
        raise ValueError("Waveform must be finite and in [-1,1]")
    if (x.square().mean(1) <= 1e-16).any():
        raise ValueError("Silent/near-silent sample: resolve QC before evaluation")


def measured_snr(x, xp):
    signal = x.double().square().mean(1)
    noise = (xp.double() - x.double()).square().mean(1)
    return 10 * torch.log10(signal / noise)


@torch.no_grad()
def at_snr(x, direction, target_db, tolerance_db=0.1):
    """Find a nonnegative scalar per sample, accounting for amplitude clipping.

    Raises instead of silently dropping samples if calibration fails.
    clip_fraction counts pre-clamp samples outside [-1,1].
    """
    _check(x)
    if direction.shape != x.shape or not torch.isfinite(direction).all():
        raise ValueError("Invalid direction")
    if not math.isfinite(target_db) or not 0 <= target_db <= 80:
        raise ValueError("Supported target SNR: 0..80 dB")
    x64 = x.double()
    u = direction.double()
    rms_u = u.square().mean(1, keepdim=True).sqrt()
    if (rms_u <= 0).any():
        raise ValueError("Zero direction for at least one sample")
    u = u / rms_u
    target_power = x64.square().mean(1, keepdim=True) / (10 ** (target_db / 10))

    def achieved_power(a):
        return ((x64 + a * u).clamp(-1, 1) - x64).square().mean(1, keepdim=True)

    lo = torch.zeros_like(target_power)
    hi = target_power.sqrt()
    for _ in range(60):
        need = achieved_power(hi) < target_power
        if not need.any():
            break
        hi = torch.where(need, hi * 2, hi)
    if (achieved_power(hi) < target_power * (1 - 1e-10)).any():
        raise ValueError("SNR target unattainable for this clipped direction")
    for _ in range(50):
        mid = (lo + hi) / 2
        low = achieved_power(mid) < target_power
        lo = torch.where(low, mid, lo)
        hi = torch.where(low, hi, mid)
    a = (lo + hi) / 2
    before_clip = x64 + a * u
    xp = before_clip.clamp(-1, 1).to(x.dtype)
    snr = measured_snr(x, xp)
    if not torch.isfinite(snr).all() or (snr - target_db).abs().max() > tolerance_db:
        raise ValueError("Measured SNR failed tolerance after dtype conversion")
    return xp, {
        "actual_snr": snr,
        "clip_fraction": (before_clip.abs() > 1).double().mean(1),
        # alpha multiplying original (unnormalised) direction
        "scale": (a / rms_u).flatten(),
    }


def fgsm_direction(model, x, y):
    """Gradient of CE with respect to waveform; all preprocessing is in model."""
    _check(x)
    xr = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        logits = model(xr)
        if logits.ndim != 2 or logits.shape[0] != x.shape[0]:
            raise ValueError("Model must return logits [B,C], also for B=1")
        loss = F.cross_entropy(logits, y.long())
        grad = torch.autograd.grad(loss, xr)[0]
    if not torch.isfinite(grad).all():
        raise ValueError("Nonfinite waveform gradient")
    if (grad.abs().sum(1) == 0).any():
        raise ValueError("Zero gradient for at least one sample")
    return grad.sign().detach()
