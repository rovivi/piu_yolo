# Etiquetado en la Mac M5 — labelImg

Workflow para corregir las 63 fotos pre-anotadas de `images/new` desde la Mac,
sin pelear con lags de X-AnyLabeling en el iGPU de Linux.

## 1. Setup (una vez)

```bash
git clone https://github.com/rovivi/piu_yolo.git   # o git pull si ya existe
cd piu_yolo
python3 -m venv venv && source venv/bin/activate
pip install labelimg
```

> Si `pip install labelimg` falla por `distutils` (Python ≥ 3.12):
> `pip install setuptools` y reintentar. Alternativa: brew `pipx`.

## 2. Abrir el editor

```bash
labelImg images/new images/new/classes.txt
```

- En el toolbar izquierdo el formato debe decir **YOLO** (clic para alternar si no).
- `classes.txt` declara las 5 clases: difficulty, fullscore, rank, score, song_name.
- Las cajas **pre-anotadas ya están en cada `.txt`** (predichas con
  `piu_ia/yolo26_v2/weights/best.pt` @ conf 0.25). Solo corregir/confirmar.

## 3. Corregir

- Borrar cajas falsas → tecla `del`.
- Agregar faltantes → `W` (rectángulo), elegir clase → guardar.
- `D` / `A` avanza/retrocede imagen; `Ctrl+S` guarda.

## 4. Commit & push

```bash
git add images/new
git commit -m "labels: correccion manual de las 63 fotos en images/new"
git push origin main
```

(En git, `git mv` no hace falta: los .txt son uno por imagen con el mismo stem.)

## 5. Integrar al dataset (en el Linux de entrenamiento)

```bash
git pull
python prepare_dataset.py add_images images/new   # copia img+txt a photos/
python normalize_names.py --apply                 # continua numeración: photo_0045.png ...
python augment_dataset.py --n 1                   # recetas equivalentes + draft augment offline
python training.py
```

## Notas

- Las imágenes nunca se renombran a mano: `normalize_names.py` renombra
  imagen + `.txt` juntos y es idempotente (se hace en Linux al integrar).
- `rename_map.csv` queda en el repo si hay que trazear un nombre viejo.
- Los caches (`labels.cache`, `*.npy`) se regeneran solos tras renombrar; no hace falta borrar nada.
