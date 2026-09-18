"""Detección de GPU centralizada: MPS (Apple Silicon M5/…) y ROCm (Linux/AMD).

Los scripts importan setup_device_env() ANTES de importar torch/ultralytics,
y detect_device() para elegir device.
"""

import os
import sys
import platform
from pathlib import Path

BASE = Path(__file__).parent


def setup_device_env():
    """Env vars de ROCm — solo en Linux. Inofensivo en Mac pero ruidoso."""
    if sys.platform != 'darwin':
        os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
        os.environ.setdefault('HSA_ENABLE_SDMA', '0')


def _fix_rocm_amdsmi():
    """torch 2.4.1+rocm6.0 + ultralytics nuevo crashea en device_count() por
    falta del módulo amdsmi. Devolvemos 1 si la GPU HIP es visible."""
    import torch
    if torch.version.hip and torch.cuda.is_available():
        torch.cuda.device_count = lambda: 1


def detect_device():
    """MPS en Apple Silicon, GPU 0 (ROCm/CUDA) en Linux, cpu como fallback."""
    if platform.system() == 'Darwin':
        import torch
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    _fix_rocm_amdsmi()
    import torch
    return 0 if torch.cuda.is_available() else 'cpu'


def amp_enabled(device):
    """AMP crash con AdamW fused en ROCm 6.0 → off. En MPS lo decide el
    check_amp interno de ultralytics (fallback automático si NaN)."""
    if platform.system() != 'Darwin':
        import torch
        if torch.version.hip:
            return False
    return device != 'cpu'


def find_best_weights():
    """best.pt más reciente entre todos los runs de piu_ia/."""
    candidates = sorted(
        BASE.glob('piu_ia/*/weights/best.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_last_checkpoint():
    """last.pt más reciente entre todos los runs de piu_ia/."""
    candidates = sorted(
        BASE.glob('piu_ia/*/weights/last.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
