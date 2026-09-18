"""Entrenamiento principal — YOLO26s sobre dataset fusionado + augmented.

v2: cos_lr, cache en disco, augment offline (data_aug.yml generado por
augment_dataset.py), mixup/copy_paste más agresivos e imgsz 1024 (batch 10
en MPS de 24 GB; a 1280 pide 25 GB y thrashea). Device auto: MPS en Apple
Silicon, ROCm/CUDA en Linux. Ver gpu.py.
"""

from gpu import setup_device_env, detect_device, amp_enabled, BASE

setup_device_env()

import platform

from ultralytics import YOLO

# Base model YOLO26 (checkpoints de YOLOv8/11 NO son compatibles — head nuevo sin DFL)
BASE_MODEL = 'yolo26s.pt'

RUN_NAME = 'yolo26_v4'

# Dataset con augmentación offline si ya fue generada; si no, el fusionado.
DATA = 'data_aug.yml' if (BASE / 'data_aug.yml').exists() else 'data_merged.yml'


def entrenar():
    device = detect_device()
    print(f"Device: {device} ({platform.system()}/{platform.machine()}) | data: {DATA}")

    model = YOLO(BASE_MODEL)

    model.train(
        data=str(BASE / DATA),
        epochs=150,
        patience=150,  # sin early stop: el anneal de cos_lr es donde se gana

        imgsz=1024,
        batch=10,
        device=device,
        amp=amp_enabled(device),
        cache='disk',
        workers=8,
        cos_lr=True,
        seed=0,

        # song_name es texto horizontal — rotación alta lo destruye
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.1,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        close_mosaic=20,
        weight_decay=0.0005,

        project='piu_ia',
        name=RUN_NAME,
        exist_ok=False,
    )

    metrics = model.val()
    print("\n=== RESULTADOS FINALES (YOLO26s v4, dataset + augmented) ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == '__main__':
    entrenar()
