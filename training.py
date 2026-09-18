import os
import sys
from pathlib import Path

# MUST be set before torch/ultralytics import (ROCm workaround, Linux/AMD only)
if sys.platform != 'darwin':
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['HSA_ENABLE_SDMA'] = '0'

import platform

from ultralytics import YOLO


def _fix_rocm_amdsmi():
    """torch 2.4.1+rocm6.0 con ultralytics nuevo crashea en device_count() por
    falta del módulo amdsmi. Nosotros devolvemos 1 si la GPU HIP es visible."""
    import torch
    if torch.version.hip and torch.cuda.is_available():
        torch.cuda.device_count = lambda: 1


BASE = Path(__file__).parent

# Base model YOLO26 (checkpoints de YOLOv8/11 NO son compatibles — head nuevo sin DFL)
BASE_MODEL = 'yolo26s.pt'

RUN_NAME = 'yolo26_v1'


def detectar_device():
    """Cross-platform device: MPS en Mac (Apple Silicon), ROCm/CUDA en Linux."""
    if platform.system() == 'Darwin':
        import torch
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    _fix_rocm_amdsmi()
    return 0  # ROCm expone API cuda en Linux/AMD


def entrenar():
    device = detectar_device()
    print(f"Device: {device} ({platform.system()}/{platform.machine()})")

    model = YOLO(BASE_MODEL)

    model.train(
        data=str(BASE / 'data_merged.yml'),
        epochs=300,
        patience=50,
        imgsz=1024,
        batch=16,
        device=device,
        amp=False,  # fused AdamW + AMP crash en ROCm 6.0 (en Mac MPS no aplica)

        # YOLO26 usa MuSGD por defecto — no forzar AdamW

        # song_name es texto horizontal — rotación alta lo destruye
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.05,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        close_mosaic=20,
        weight_decay=0.0005,
        copy_paste=0.1,

        project='piu_ia',
        name=RUN_NAME,
        exist_ok=False,
    )

    metrics = model.val()
    print("\n=== RESULTADOS FINALES (YOLO26s, dataset fusionado) ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == '__main__':
    entrenar()
