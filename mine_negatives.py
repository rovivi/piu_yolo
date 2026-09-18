"""
Hard negative mining: corre el modelo en imágenes sin anotar, muestra detecciones,
te deja marcar cuáles son falsos positivos y los agrega al dataset.

Uso:
  python mine_negatives.py /carpeta/con/imagenes [--conf 0.5] [--model piu_ia/yolo26_v1/weights/best.pt]
  python mine_negatives.py /carpeta/con/imagenes --auto  # agrega todo como negativo (sin revisión manual)

Con --auto: ideal si YA SABES que esas imágenes no tienen objetos válidos.
Sin --auto: abre cada imagen en una ventana, presiona:
  y = falso positivo → agregar como negativo
  n = correcto, no agregar
  q = salir
"""

import sys
import shutil
import argparse
from pathlib import Path

from gpu import setup_device_env, find_best_weights

setup_device_env()

BASE = Path(__file__).parent
NEGATIVES_OUT = BASE / "negatives_mined"

CLASSES = {0: "difficulty", 1: "fullscore", 2: "rank", 3: "score", 4: "song_name"}

IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def find_images(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)


def mine(source_dir: str, model_path: str, conf: float, auto: bool):
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print("Instala: pip install ultralytics opencv-python")
        sys.exit(1)

    src = Path(source_dir)
    if model_path is None:
        print("No hay ningún best.pt en piu_ia/ — usa --model para especificar uno")
        sys.exit(1)
    model_p = Path(model_path)
    if not model_p.exists():
        print(f"Modelo no encontrado: {model_p}")
        print("Usa --model para especificar la ruta al .pt")
        sys.exit(1)

    imgs = find_images(src)
    if not imgs:
        print(f"No hay imágenes en {source_dir}")
        sys.exit(1)

    NEGATIVES_OUT.mkdir(exist_ok=True)
    model = YOLO(str(model_p))

    marked_as_negative = []

    for img_path in imgs:
        results = model(str(img_path), conf=conf, verbose=False)
        detections = results[0].boxes

        if len(detections) == 0:
            if auto:
                # Sin detecciones = imagen limpia → buen negativo también
                dst = NEGATIVES_OUT / img_path.name
                if not dst.exists():
                    shutil.copy2(str(img_path), dst)
                    marked_as_negative.append(img_path)
            continue

        if auto:
            dst = NEGATIVES_OUT / img_path.name
            if not dst.exists():
                shutil.copy2(str(img_path), dst)
                marked_as_negative.append(img_path)
            print(f"  [auto-neg] {img_path.name} — {len(detections)} detecciones")
            continue

        # Modo manual: muestra la imagen con bboxes
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            label = f"{CLASSES.get(cls_id, cls_id)} {conf_val:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display = img.copy()
        # Escala para que entre en pantalla
        max_dim = 900
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        cv2.imshow(f"Mine — {img_path.name} | y=FP(agregar neg) n=skip q=salir", display)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord('q'):
            print("Saliendo.")
            break
        elif key == ord('y'):
            dst = NEGATIVES_OUT / img_path.name
            shutil.copy2(str(img_path), dst)
            marked_as_negative.append(img_path)
            print(f"  [FP] {img_path.name}")
        else:
            print(f"  [skip] {img_path.name}")

    print(f"\n{len(marked_as_negative)} imágenes marcadas como negativas → {NEGATIVES_OUT}")
    if marked_as_negative:
        print("\nAhora ejecuta para agregarlas al dataset:")
        print(f"  python prepare_dataset.py add_negatives {NEGATIVES_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="Carpeta con imágenes a revisar")
    parser.add_argument("--model", default=None, help="Ruta al best.pt (default: el más nuevo)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confianza mínima (default 0.3)")
    parser.add_argument("--auto", action="store_true",
                        help="Agrega todas como negativas sin revisión manual")
    args = parser.parse_args()

    model_path = args.model or find_best_weights()
    mine(args.source_dir, model_path, args.conf, args.auto)
