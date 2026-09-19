"""Data augmentation offline del split de train: degradaciones fotorrealistas
de fotos de cabina (lo que destruye la detección en el teléfono) + perspectiva.

Genera `augmented/images/train` + `augmented/labels/train` con 1 variante por
imagen de train (frames + fotos), y `data_aug.yml` que lista originals + augment.
El split val NUNCA se toca — los benchmarks siguen siendo comparables.

Cada variante aplica de forma aleatoria (seed fija):
  - perspectiva ±8° (homografía por esquinas, cajas transformadas exacto)
  - blur de movimiento (tiembla la mano) o defocus
  - rescale down/up (screenshot/compresión)
  - JPEG q30-70
  - brillo/gamma/contraste (cabina oscura) + parche de glare
  - ruido gaussiano
Las cajas vacías (negativos) quedan vacías.

Uso:
  python augment_dataset.py            # 1 variante por imagen de train
  python augment_dataset.py --n 2      # 2 variantes por imagen
  python augment_dataset.py --clean    # borra augmented/ y regenera
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent
IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
SPLITS = [(BASE / 'images/train', BASE / 'labels/train'),
          (BASE / 'photos/images/train', BASE / 'photos/labels/train')]
OUT_IMG = BASE / 'augmented/images/train'
OUT_LBL = BASE / 'augmented/labels/train'


def load_boxes(lbl: Path, w: int, h: int):
    boxes = []
    for line in lbl.read_text().strip().splitlines() if lbl.exists() else []:
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        x, y, bw, bh = (float(v) for v in p[1:5])
        boxes.append([c, x * w, y * h, bw * w, bh * h])
    return boxes


def save_boxes(boxes, lbl: Path, w: int, h: int):
    lines = []
    for c, cx, cy, bw, bh in boxes:
        lines.append(f"{c} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    lbl.write_text('\n'.join(lines))


def warp_perspective(img, boxes, max_deg=8.0):
    """Homografía aleatoria (esquina a esquina), cajas por punto central+extremos."""
    h, w = img.shape[:2]
    max_shift = int(min(w, h) * max_deg / 90 * 1.6)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[np.random.uniform(-max_shift, 0), np.random.uniform(-max_shift, 0)],
                            [np.random.uniform(0, max_shift), np.random.uniform(-max_shift, 0)],
                            [np.random.uniform(0, max_shift), np.random.uniform(0, max_shift)],
                            [np.random.uniform(-max_shift, 0), np.random.uniform(0, max_shift)]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    new_boxes = []
    for c, cx, cy, bw, bh in boxes:
        pts = np.float32([[[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                           [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]]])
        pts = cv2.perspectiveTransform(pts, M)[0]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        nw, nh = x1 - x0, y1 - y0
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        # clip al lienzo; descartar cajas que quedan casi afuera
        xc = min(max(xc, 0), w)
        yc = min(max(yc, 0), h)
        bw = min(bw if nw <= 0 else nw, w)
        bh = min(bh if nh <= 0 else nh, h)
        if bw < 8 or bh < 8:
            continue
        x0c, y0c = max(xc - bw / 2, 0), max(yc - bh / 2, 0)
        x1c, y1c = min(xc + bw / 2, w), min(yc + bh / 2, h)
        new_boxes.append([c, (x0c + x1c) / 2, (y0c + y1c) / 2, x1c - x0c, y1c - y0c])
    return out, new_boxes


def motion_blur(img):
    k = np.random.choice([7, 11, 15])
    angle = np.random.uniform(0, 180)
    kern = np.zeros((k, k), np.float32)
    cv2.line(kern, (k // 2, k // 2), (int(k // 2 + np.cos(np.deg2rad(angle)) * k // 2),
                                      int(k // 2 + np.sin(np.deg2rad(angle)) * k // 2)), 1.0)
    s = kern.sum()
    return cv2.filter2D(img, -1, kern / s if s > 0 else kern) if s > 0 else img


def defocus_blur(img):
    return cv2.GaussianBlur(img, (0, 0), np.random.uniform(1.5, 3.5))


def lowres(img):
    h, w = img.shape[:2]
    f = np.random.uniform(2.0, 4.0)
    small = cv2.resize(img, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def photometric(img):
    """Brillo/gamma/contraste para cabina oscura y sobreexpuesta."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    out *= np.random.uniform(0.4, 1.45)                      # brillo
    out = 255 * np.power(np.clip(out / 255, 0, 1), np.random.uniform(0.6, 1.8))  # gamma
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * np.random.uniform(0.8, 1.25) + mean  # contraste
    out = np.clip(out, 0, 255).astype(np.uint8)

    if np.random.rand() < 0.45:                              # glare: blob elíptico
        glare = np.zeros((h, w), np.float32)
        gr, gc, gax, gay = (np.random.randint(0, w), np.random.randint(0, h),
                            np.random.randint(w // 8, w // 3), np.random.randint(h // 8, h // 3))
        cv2.ellipse(glare, (gr, gc), (gax, gay), np.random.uniform(0, 180), 0, 360, 1, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), min(gax, gay) / 2)
        glare = (glare / max(glare.max(), 1e-6)) * np.random.uniform(60, 140)
        out = np.clip(out.astype(np.float32) + glare[..., None], 0, 255).astype(np.uint8)
    return out


def jpeg_noise(img):
    q = np.random.randint(25, 71)
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
    if img is None:
        return None
    if np.random.rand() < 0.5:
        sigma = np.random.uniform(4, 14)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def augment(img, boxes):
    h, w = img.shape[:2]
    if np.random.rand() < 0.6:
        img, boxes = warp_perspective(img, boxes)
    if np.random.rand() < 0.5:
        img = motion_blur(img) if np.random.rand() < 0.6 else defocus_blur(img)
    if np.random.rand() < 0.4:
        img = lowres(img)
    img = photometric(img)
    img = jpeg_noise(img)
    if np.random.rand() < 0.3:  # jitter HSV (neones de colores desbordan)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = np.clip(hsv[..., 0].astype(np.int16) + np.random.randint(-8, 9), 0, 179)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.int16) + np.random.randint(-40, 20), 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1, help='variantes por imagen de train')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.clean and (BASE / 'augmented').exists():
        shutil.rmtree(BASE / 'augmented')

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

    manifest = []
    total = 0
    for idir, ldir in SPLITS:
        imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
        for im in imgs:
            lbl = ldir / f'{im.stem}.txt'
            img = cv2.imread(str(im))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes = load_boxes(lbl, w, h)
            for v in range(args.n):
                name = f'aug{v}_{im.stem}'
                out_img = OUT_IMG / f'{name}.jpg'
                out_lbl = OUT_LBL / f'{name}.txt'
                if out_img.exists() and out_lbl.exists():
                    manifest.append({'src': im.name, 'out': name,
                                     'n_boxes': len(out_lbl.read_text().splitlines())})
                    total += 1
                    continue
                np.random.seed(args.seed + total)  # determinista por imagen
                aug_img, aug_boxes = augment(img.copy(), [list(b) for b in boxes])
                if aug_img is None:
                    continue
                cv2.imwrite(str(out_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                save_boxes(aug_boxes, out_lbl, w, h)
                manifest.append({'src': im.name, 'out': name, 'n_boxes': len(aug_boxes)})
                total += 1

    (BASE / 'augmented/manifest.json').write_text(json.dumps(manifest, indent=2))
    yml = f"""# Train = originales + augmented/ (degradaciones fotorrealistas offline).
# Val intocado — benchmarks comparables contra baseline_yolo26_v1.
train:
  - images/train
  - photos/images/train
  - augmented/images/train

val:
  - images/val
  - photos/images/val

names:
  0: difficulty
  1: fullscore
  2: rank
  3: score
  4: song_name
"""
    (BASE / 'data_aug.yml').write_text(yml)
    print(f"OK: {total} imágenes augmentadas en augmented/ (+{total} train). data_aug.yml generado.")


if __name__ == '__main__':
    main()
