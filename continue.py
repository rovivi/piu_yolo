"""Reanuda el último entrenamiento interrumpido (busca el last.pt más nuevo).

Uso:
  python continue.py                # resume el run más reciente de piu_ia/
  python continue.py --model ruta/last.pt
"""

import argparse

from gpu import setup_device_env, find_last_checkpoint

setup_device_env()

from ultralytics import YOLO


def resumir(model_path):
    if not model_path.exists():
        print(f"❌ No hay checkpoint: {model_path}")
        print("Lanza un entrenamiento primero: python training.py")
        return

    print(f"✅ Resumiendo desde {model_path}")
    model = YOLO(str(model_path))
    model.train(resume=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='Ruta a last.pt (default: el más nuevo)')
    args = parser.parse_args()

    path = args.model or find_last_checkpoint()
    resumir(path)
