"""
Prepara el dataset: crea split train/val real y agrega imágenes negativas.

Uso:
  python prepare_dataset.py split          # split 80/20 train/val
  python prepare_dataset.py add_negatives /carpeta/con/falsos_positivos
  python prepare_dataset.py add_images /carpeta/con/nuevas_imagenes_anotadas
  python prepare_dataset.py status         # muestra estadísticas del dataset
"""

import os
import shutil
import random
import argparse
from pathlib import Path

BASE = Path(__file__).parent
IMAGES = BASE / "images"
LABELS = BASE / "labels"
IMAGES_TRAIN = BASE / "images" / "train"
IMAGES_VAL   = BASE / "images" / "val"
LABELS_TRAIN = BASE / "labels" / "train"
LABELS_VAL   = BASE / "labels" / "val"

DATA_YML = BASE / "data.yml"

IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def find_images(folder: Path):
    """Case-insensitive: en macOS los globs *.jpg y *.JPG matchean lo mismo
    (filesystem case-insensitive) y duplicaban la lista."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)


def status():
    if IMAGES_TRAIN.exists():
        train_imgs = find_images(IMAGES_TRAIN)
        val_imgs   = find_images(IMAGES_VAL)
        train_negs = sum(1 for f in LABELS_TRAIN.glob("*.txt") if f.stat().st_size == 0)
        val_negs   = sum(1 for f in LABELS_VAL.glob("*.txt") if f.stat().st_size == 0)
        print(f"Dataset con split:")
        print(f"  train: {len(train_imgs)} imágenes ({train_negs} negativas)")
        print(f"  val:   {len(val_imgs)} imágenes ({val_negs} negativas)")
    else:
        imgs = find_images(IMAGES)
        lbls = list(LABELS.glob("*.txt"))
        negs = sum(1 for f in lbls if f.stat().st_size == 0)
        print(f"Dataset flat (sin split):")
        print(f"  {len(imgs)} imágenes, {len(lbls)} labels ({negs} negativas/vacías)")


def split(val_ratio=0.2, seed=42):
    """Mueve imágenes a train/ y val/ si aún están en flat."""
    imgs = sorted(find_images(IMAGES))
    imgs = [p for p in imgs if p.parent == IMAGES]  # solo las del root

    if not imgs:
        print("No hay imágenes flat en images/ — ya está spliteado o carpeta vacía.")
        return

    random.seed(seed)
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    val_set   = set(p.stem for p in imgs[:n_val])
    train_set = set(p.stem for p in imgs[n_val:])

    for folder in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        folder.mkdir(parents=True, exist_ok=True)

    moved_train = moved_val = 0
    for img_path in imgs:
        stem = img_path.stem
        lbl_path = LABELS / f"{stem}.txt"
        if stem in val_set:
            shutil.move(str(img_path), IMAGES_VAL / img_path.name)
            if lbl_path.exists():
                shutil.move(str(lbl_path), LABELS_VAL / lbl_path.name)
            else:
                (LABELS_VAL / f"{stem}.txt").touch()
            moved_val += 1
        else:
            shutil.move(str(img_path), IMAGES_TRAIN / img_path.name)
            if lbl_path.exists():
                shutil.move(str(lbl_path), LABELS_TRAIN / lbl_path.name)
            else:
                (LABELS_TRAIN / f"{stem}.txt").touch()
            moved_train += 1

    # Actualiza data.yml — sin `path`: las rutas relativas se resuelven contra
    # el directorio del propio YAML (ultralytics ≥8.3 resuelve `path:` relativo
    # contra DATASETS_DIR global, así que omitirlo es lo portable)
    yml_content = f"""# Dataset de frames de video. Rutas relativas a este .yml.
train: images/train
val: images/val

names:
  0: difficulty
  1: fullscore
  2: rank
  3: score
  4: song_name
"""
    DATA_YML.write_text(yml_content)
    print(f"Split listo: {moved_train} train, {moved_val} val")
    print(f"data.yml actualizado con train/val paths.")


def add_negatives(source_dir: str):
    """
    Agrega imágenes de falsas positivos como negativas al dataset de train.
    Crea label vacío para cada imagen → YOLO aprende que esas regiones son fondo.
    """
    src = Path(source_dir)
    if not src.exists():
        print(f"No existe: {source_dir}")
        return

    imgs = find_images(src)
    if not imgs:
        print("No hay imágenes en esa carpeta.")
        return

    # Decide destino según si ya hay split o no
    if IMAGES_TRAIN.exists():
        dest_imgs = IMAGES_TRAIN
        dest_lbls = LABELS_TRAIN
    else:
        dest_imgs = IMAGES
        dest_lbls = LABELS

    dest_imgs.mkdir(parents=True, exist_ok=True)
    dest_lbls.mkdir(parents=True, exist_ok=True)

    added = 0
    for img in imgs:
        new_name = f"neg_{img.name}"
        dst_img = dest_imgs / new_name
        dst_lbl = dest_lbls / f"neg_{img.stem}.txt"

        if dst_img.exists():
            print(f"  Skip (ya existe): {new_name}")
            continue

        shutil.copy2(str(img), dst_img)
        dst_lbl.touch()  # label vacío = negativo puro
        added += 1

    print(f"Agregadas {added} imágenes negativas a {dest_imgs}")
    print("Label vacío = YOLO trata esas zonas como fondo.")


def add_images(source_dir: str):
    """
    Agrega nuevas imágenes anotadas. Busca pares imagen+label (.txt YOLO).
    Si no hay label para una imagen, la agrega como negativa.
    """
    src = Path(source_dir)
    if not src.exists():
        print(f"No existe: {source_dir}")
        return

    imgs = find_images(src)

    if IMAGES_TRAIN.exists():
        dest_imgs = IMAGES_TRAIN
        dest_lbls = LABELS_TRAIN
    else:
        dest_imgs = IMAGES
        dest_lbls = LABELS

    dest_imgs.mkdir(parents=True, exist_ok=True)
    dest_lbls.mkdir(parents=True, exist_ok=True)

    added_pos = added_neg = skipped = 0
    for img in imgs:
        lbl = img.parent / f"{img.stem}.txt"
        dst_img = dest_imgs / img.name
        dst_lbl = dest_lbls / f"{img.stem}.txt"

        if dst_img.exists():
            skipped += 1
            continue

        shutil.copy2(str(img), dst_img)
        if lbl.exists() and lbl.stat().st_size > 0:
            shutil.copy2(str(lbl), dst_lbl)
            added_pos += 1
        else:
            dst_lbl.touch()
            added_neg += 1

    print(f"Agregadas: {added_pos} anotadas, {added_neg} negativas, {skipped} skipped (ya existían)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("split")
    sub.add_parser("status")

    p_neg = sub.add_parser("add_negatives")
    p_neg.add_argument("source_dir")

    p_img = sub.add_parser("add_images")
    p_img.add_argument("source_dir")

    args = parser.parse_args()

    if args.cmd == "split":
        split()
    elif args.cmd == "add_negatives":
        add_negatives(args.source_dir)
    elif args.cmd == "add_images":
        add_images(args.source_dir)
    elif args.cmd == "status":
        status()
    else:
        parser.print_help()
