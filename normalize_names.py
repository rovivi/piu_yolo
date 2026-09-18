"""
Normaliza nombres de imágenes + labels en pares (no rompe el etiquetado).

Renombra imagen y su .txt juntos a numeración secuencial por familia:
  - frames de video:  frame_0001.jpg  + frame_0001.txt
  - fotos de teléfono: photo_0001.jpg + photo_0001.txt

Determinista (orden alfabético del nombre actual) e idempotente: los nombres
nuevos se ejecutan dos veces y quedan igual. Genera rename_map.csv con el
mapeo viejo→nuevo para trazabilidad, y borra etiquetas ultralytics.

Uso:
  python normalize_names.py            # dry-run: muestra qué hará
  python normalize_names.py --apply    # ejecuta los renombres
"""

import csv
import shutil
import argparse
from pathlib import Path

BASE = Path(__file__).parent
IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# familia → [(dir_imágenes, dir_labels)]
FAMILIES = {
    "frame": [(BASE / "images" / "train", BASE / "labels" / "train"),
              (BASE / "images" / "val",   BASE / "labels" / "val"),
              (BASE / "images",           BASE / "labels")],
    "photo": [(BASE / "photos" / "images" / "train", BASE / "photos" / "labels" / "train"),
              (BASE / "photos" / "images" / "val",   BASE / "photos" / "labels" / "val")],
}

STALE_CACHES = [BASE / "labels.cache"]


def find_images(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)


def build_plan():
    """Devuelve lista de (path_viejo, path_nuevo, path_label_viejo, path_label_nuevo)."""
    plan = []
    counters = {fam: 1 for fam in FAMILIES}

    def next_name(fam, stem):
        n = counters[fam]
        counters[fam] += 1
        return f"{fam}_{n:04d}"

    taken = set()
    fam_moves = {fam: [] for fam in FAMILIES}

    # Pasada 1: registra los que ya tienen la forma final
    for fam, dirs in FAMILIES.items():
        for dir_imgs, dir_lbls in dirs:
            for img in find_images(dir_imgs):
                if img.stem.split('_')[0] == fam and not img.stem.startswith(("neg_", "aug_")):
                    counters[fam] = max(counters[fam], int(img.stem.split('_')[1]) + 1) \
                        if img.stem.split('_')[1].isdigit() else counters[fam]
                    taken.add(img.stem)

    # Pasada 2: plan para los que hay que renombrar
    for fam, dirs in FAMILIES.items():
        for dir_imgs, dir_lbls in dirs:
            for img in find_images(dir_imgs):
                already = (img.stem in taken) and img.stem.count('_') == 1 \
                    and img.stem.split('_')[1].isdigit()
                if already:
                    continue
                new_stem = next_name(fam, img.stem)
                while new_stem in taken:            # no aplastar nada existente
                    new_stem = next_name(fam, img.stem)
                taken.add(new_stem)
                lbl_path = dir_lbls / f"{img.stem}.txt"
                plan.append((img, dir_imgs / f"{new_stem}{img.suffix}", lbl_path,
                             dir_lbls / f"{new_stem}.txt", fam))
                fam_moves[fam].append(img.name)

    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    plan = build_plan()
    print(f"Pares a renombrar: {len(plan)}")
    for img, new_img, lbl, new_lbl, fam in plan:
        print(f"  [{fam}] {img.name} + {lbl.name}  ->  {new_img.name} + {new_lbl.name}")

    if not args.apply:
        print("\nDry-run. Ejecuta con --apply para renombrar.")
        return

    if not plan:
        return

    with open(BASE / "rename_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "old_image", "new_image", "old_label", "new_label"])
        for img, new_img, lbl, new_lbl, fam in plan:
            w.writerow([fam, img.name, new_img.name, lbl.name, new_lbl.name])

    thumbs = 0
    for img, new_img, lbl, new_lbl, fam in plan:
        shutil.move(str(img), str(new_img))
        if lbl.exists():
            shutil.move(str(lbl), str(new_lbl))
        thumbs += 1

    for c in STALE_CACHES:
        if c.exists():
            c.unlink()

    print(f"Renombrados {thumbs} pares. Mapa en rename_map.csv")
    print("Caches ultralytics invalidados; el próximo train re-escaneará el dataset.")


if __name__ == "__main__":
    main()
