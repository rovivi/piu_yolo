# PIU YOLO Training 💃🕹️

[![Modelo YOLO26](https://img.shields.io/badge/modelo-YOLO26s-yellow)](https://docs.ultralytics.com/models/yolo26)
[![Dataset fusionado](https://img.shields.io/badge/dataset-204%20train%20%2F%2054%20val-blue)](#-datasets)
[![Val mAP50](https://img.shields.io/badge/val_mAP50-0.826-brightgreen)](#resultados-de-validación)

Detector de objetos con **YOLO26** (Ultralytics) entrenado para leer la interfaz del juego **Pump It Up (PIU)**: nombres de canciones, puntajes, rangos y dificultad, tanto de **video de gameplay** como de **fotos de cabina/teléfono**.

```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Gameplay /  │    │  YOLO26s      │    │ Pipeline de OCR  │    │ Estadísticas │
│  fotos de    │──▶ │  piu_ia/      │──▶ │ (score/rank/     │──▶ │ automáticas, │
│  teléfonos   │    │  yolo26_v1    │    │  song parsing)   │    │  overlays    │
└──────────────┘    └───────────────┘    └──────────────────┘    └──────────────┘
```

---

## 🚀 Quickstart

```bash
# 1. Clonar (el dataset ya está dentro del repo — funciona en Linux y Mac)
git clone https://github.com/rovivi/piu_yolo.git
cd piu_yolo

# 2. Entorno virtual
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Validar GPU (opcional)
python check_gpu.py

# 4. Entrenar (detecta device: MPS en Mac M5, ROCm/CUDA en Linux)
python training.py          # o: ./train.sh
./train.sh --resume         # continuar el último run interrumpido
```

Los pesos finales quedan en `piu_ia/yolo26_v1/weights/best.pt`.

---

## 📂 Estructura del Proyecto

```
piu_yolo/
├── training.py          # ⭐ Entrenamiento principal (YOLO26s, device auto)
├── continue.py          # Resume el run más reciente (busca el last.pt nuevo solo)
├── gpu.py               # ⭐ Detección device (MPS/ROCm/CUDA) + env ROCm centralizado
├── mine_negatives.py    # Hard-negative mining interactivo
├── prepare_dataset.py   # Split train/val + gestión de negativos
├── check_gpu.py         # Diagnóstico multiplataforma (MPS/ROCm/CUDA/CPU)
│
├── data.yml             # Dataset de frames de video (160 train / 40 val)
├── data_merged.yml      # ⭐ Dataset fusionado (video + fotos), portable
├── requirements.txt     # Dependencias (ultralytics + opencv)
├── train.sh             # Lanzador portable (venv/conda/python3, --resume)
├── images/labels/       # Frames de video + labels YOLO
├── photos/              # ✨ Fotos de cabina/teléfono (44 train / 14 val)
├── piu_ia/              # Runs de entrenamiento (checkpoints + args.yaml)
├── datasets/coco8/      # Fixture mini de test (ultralytics)
├── test_best_model/     # App standalone de inferencia + build PyInstaller
└── yolo26s.pt           # Base model YOLO26 Small
```

## 🏷️ Clases Detectadas

| ID | Clase | Descripción |
|---|---|---|
| 0 | `difficulty` | Nivel de dificultad |
| 1 | `fullscore` | Puntaje máximo posible/acumulado |
| 2 | `rank` | Grado S/SS/A… |
| 3 | `score` | Puntaje obtenido |
| 4 | `song_name` | Título de la canción |

## 📊 Resultados de Validación

Dataset fusionado (204 train / 54 val), val @ 1024px (val **intocado** en todos los runs — benchmarks comparables):

| Run | Receta | mAP50 | mAP50-95 |
|---|---|---|---|
| `v5_photos` | YOLOv8n (cadena finetune) | 0.811 | 0.424 |
| `yolo26_v1` | YOLO26s desde scratch, sin aug offline | 0.826 | 0.424 |
| `yolo26_v2` | + aug offline, mixup 0.15, early-stop ep92 | 0.814 | **0.439** |
| `yolo26_v3` | + aug offline, mixup 0.05, early-stop ep36 | 0.806 | 0.420 |
| `yolo26_v4` | v3 sin early-stop (anneal completo de cos_lr) | — *en curso* | — |

Por clase (val @1024 — mAP50, con delta vs `yolo26_v1`):

| Clase | v1 | v2 | v3 |
|---|---|---|---|
| score | 0.938 | 0.929 (-1) | 0.870 |
| rank | 0.905 | 0.884 (-2) | 0.857 |
| fullscore | 0.86 | 0.798 (-6) | 0.830 |
| difficulty | 0.813 | 0.793 (-2) | 0.771 |
| **song_name** ⚠️ | 0.613 | 0.668 (**+5.5**, R 0.558→0.693) | **0.700 (+8.7)** |

> **Hallazgos del ciclo v2–v4 (Mac M5 Pro, 24 GB):**
> - La **augmentación offline** (`augment_dataset.py`: perspectiva ±8°, motion/defocus blur, low-res, JPEG q25-70, glare, gamma) sube `song_name` **+5.5 a +8.7 pts** y su recall **+13 pts** — la clase débil deja de ser tan débil.
> - El **early-stop con val de 54 imágenes mata el rendimiento**: la patience dispara antes del anneal de `cos_lr` (v2 best ep92, v3 best ep36, ambos sin fase final). v4 = v3 con `patience=epochs` para forzar anneal completo — es el candidato a beat.
> - `fullscore` y el mAP50 global bajan ~1-6 pts con mixup/copy_paste agresivos (0.15): quedaron en 0.05/0.1 (valores v1).
> - A imgsz 1280 con batch 12 MPS pide 25.6 GB > 24 GB → thrash. En la Mac: **batch 10 @ 1024** es el techo (14.6 GB).
>
> Benchmarks detallados en `benchmarks/*.json`, reproducibles con `python benchmark.py --compare benchmarks/baseline_yolo26_v1.json`.

> `song_name` es la clase débil: texto largo y variable, con pocas imágenes. Prioridad de mejora #1.

## 🏋️ Entrenamiento

`training.py` está configurado con (receta v4):

- **Modelo**: `yolo26s.pt` (YOLO26 = NMS-free end-to-end, sin DFL, optimizer auto)
- **Dataset**: `data_aug.yml` (generado por `augment_dataset.py`) = originales + `augmented/`; fallback `data_merged.yml`
- **150 épocas**, `patience=150` (sin early-stop: el anneal de `cos_lr` es donde se gana), `imgsz=1024`, batch 10 (MPS 24 GB) / 16 (ROCm)
- **Augmentations**: rotation 5°, sin shear/flips (texto horizontal), mosaic 1.0 con `close_mosaic=20`, mixup 0.05, copy_paste 0.1, `cos_lr=True`, `cache=disk`
- **Offline**: `python augment_dataset.py --n 1` genera 204 variantes degradadas (perspectiva/blur/lowres/jpeg/glare) — correr UNA vez, es determinista (seed 42)

```bash
python training.py              # Entrena YOLO26s con dataset fusionado
./train.sh --resume             # Resume desde checkpoint interrumpido
```

## 🔴 Hard Negative Mining

Workflow para matar falsos positivos en fotos reales:

```bash
# Interactivo: muestra detecciones, marca (y/n) si son FP
python mine_negatives.py /carpeta/imagenes --model piu_ia/yolo26_v1/weights/best.pt

# Automático: agrega todo como negativo (solo si sabes que no hay objetos válidos)
python mine_negatives.py /carpeta/imagenes --auto
```

Después integra los negativos con:

```bash
python prepare_dataset.py add_negatives negatives_mined
python prepare_dataset.py split      # regenera el split 80/20
python prepare_dataset.py status     # estadísticas actuales del dataset
```

## 🛠️ Requisitos Multiplataforma

| Plataforma | Device | Notes |
|---|---|---|
| **Mac Apple Silicon (M5)** | `mps` (auto) | AMP se activa — el check interno de ultralytics hace fallback solo si da NaN. Detección centralizada en `gpu.py`. |
| **Linux AMD ROCm** | `0` (auto) | torch 2.4.1+rocm6.0 en Ubuntu tiene 2 bugs conocidos: `amdsmi` (NameError) y el crash de AdamW fused con AMP (`amp=False` automático vía `gpu.py`). Workarounds incluidos. |
| Linux/CUDA, CPU | auto | Sin configuración extra |

Detalles en [sección de troubleshooting](#troubleshooting).

## 📦 Export para apps (Android / iOS)

```python
from ultralytics import YOLO

# YOLO26 exporta más limpio que v8/11 — NMS-free elimina ops custom problemáticas
YOLO("piu_ia/yolo26_v1/weights/best.pt").export(format="ncnn")    # Android (también tflite)
YOLO("piu_ia/yolo26_v1/weights/best.pt").export(format="coreml")  # iOS (Metal/ANE)
YOLO("piu_ia/yolo26_v1/weights/best.pt").export(format="openvino")# Intel/OpenVINO
```

Apps de referencia de la app existente: `test_best_model/` (PyInstaller) y el repo `piu_monolith` (export OpenVINO/NCNN/OpenVINO).

## 🐞 Troubleshooting

| Síntoma | Causa | Fix |
|---|---|---|
| `NameError: name 'amdsmi' is not defined` en torch ROCm | torch 2.4.1+rocm6.0 sin módulo amdsmi | `sed -i 's/except amdsmi\.AmdSmiException as e:/except Exception as e:/g' venv/lib64/python3.12/site-packages/torch/cuda/__init__.py` |
| `RuntimeError: params, grads ... must have same dtype` al iniciar entrenamiento | AdamW fused + AMP broken en ROCm 6.0 | `amp=False` en `model.train()` (ya en training.py) |
| GPU no usada / `HSA` errors | iGPU AMD GFX 10.3 no soportada | `HSA_OVERRIDE_GFX_VERSION=10.3.0` y `HSA_ENABLE_SDMA=0` (seteenados en training.py solo en Linux). Ver `check_gpu.py` |
| Run se guarda en `runs/detect/piu_ia/...` | ultralytics ≥8.4 redirige `project` bajo `runs/` | Copiar weights a `piu_ia/<run>/weights/` |
| Weights de runs viejos no cargan en YOLO26 | Head nuevo sin DFL — incompatibles | Retraine desde `yolo26s.pt`. No intentar finetune de checkpoints v8/v11 |

## 📜 Referencias

- [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26)
- [YOLOv8 vs YOLO26 Comparison](https://docs.ultralytics.com/compare/yolov8-vs-yolo26)
- Repo de inferencia offline: `piu_monolith`

---
*Desarrollado para la comunidad de Pump It Up.*
