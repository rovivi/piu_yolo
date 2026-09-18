import os
from pathlib import Path

# MUST be set before torch/ultralytics import
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HSA_ENABLE_SDMA'] = '0'

from ultralytics import YOLO

BASE = Path(__file__).parent
BEST_PT = BASE / "piu_ia" / "improved_v2_adamw" / "weights" / "best.pt"


BEST_V3 = BASE / "piu_ia" / "v3_negatives3" / "weights" / "best.pt"

def entrenar(finetune=True):
    if finetune and BEST_V3.exists():
        model = YOLO(str(BEST_V3))
        run_name = "v4_finetune"
        epochs = 400
        lr0 = 0.0002   # LR bajo — no destruir pesos
        print(f"Fine-tune desde: {BEST_V3}")
    else:
        model = YOLO('yolov8n.pt')
        run_name = "v4_scratch"
        epochs = 700
        lr0 = 0.001
        print("Entrenando desde yolov8n.pt base")

    model.train(
        data='data.yml',
        epochs=epochs,
        patience=100,   # más paciencia — el anterior cortó demasiado pronto
        imgsz=1024,
        batch=16,
        device=0,

        optimizer='AdamW',
        lr0=lr0,
        lrf=0.01,
        cos_lr=True,

        # song_name es texto horizontal — rotación alta lo destruye
        degrees=5.0,    # antes 15 → bajado a 5
        translate=0.1,
        scale=0.5,
        shear=0.0,      # antes 0.5 → 0 (deforma texto)
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.05,     # reducido — menos confusión entre clases

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        close_mosaic=20,
        weight_decay=0.0005,
        copy_paste=0.1,  # augmentation extra para clases difíciles

        project='piu_ia',
        name=run_name,
        exist_ok=False,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-scratch', action='store_true',
                        help='Ignora best.pt y entrena desde yolov8n base')
    args = parser.parse_args()
    entrenar(finetune=not args.from_scratch)
