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
pip install ultralytics

# 3. Validar GPU (opcional)
python check_gpu.py

# 4. Entrenar (detecta device: MPS en Mac, ROCm/CUDA en Linux)
python training.py
```

Los pesos finales quedan en `piu_ia/yolo26_v1/weights/best.pt`.

---

## 📂 Estructura del Proyecto

```
piu_yolo/
├── training.py          # ⭐ Entrenamiento principal (YOLO26s, device auto)
├── continue.py          # Reanudar entrenamiento interrumpido
├── mine_negatives.py    # Hard-negative mining interactivo
├── prepare_dataset.py   # Split train/val + gestión de negativos
├── check_gpu.py         # Diagnóstico de ROCm/CUDA
├── simple_check.py      # Sanity check de torch
│
├── data.yml             # Dataset de frames de video (160 train / 40 val)
├── data_merged.yml      # ⭐ Dataset fusionado (video + fotos), portable
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

Dataset fusionado (204 train / 54 val), val @ 1024px:

| Run | Modelo | mAP50 | mAP50-95 |
|---|---|---|---|
| `v5_photos` | YOLOv8n (cadena finetune) | 0.811 | 0.424 |
| `yolo26_v1` ✅ actual | YOLO26s desde scratch | **0.826** | **0.425** |

Por clase (`yolo26_v1`, val final — mAP50 / P / R):

| Clase | mAP50 | Precision | Recall |
|---|---|---|---|
| score | 0.938 | 0.877 | 0.905 |
| rank | 0.905 | 0.861 | 0.816 |
| fullscore | 0.86 | 0.849 | 0.804 |
| difficulty | 0.813 | 0.855 | 0.837 |
| **song_name** ⚠️ | 0.615 | 0.677 | 0.557 |

> `song_name` es la clase débil: texto largo y variable, con pocas imágenes. Prioridad de mejora #1.

## 🏋️ Entrenamiento

`training.py` está configurado con:

- **Modelo**: `yolo26s.pt` (YOLO26 = NMS-free end-to-end, sin DFL, optimizer MuSGD automático)
- **Dataset**: `data_merged.yml` — rutas **relativas** al YAML, portable entre máquinas
- **300 épocas**, `patience=50`, `imgsz=1024`, batch 16
- **Augmentations pensados para PIU**: rotation 5° (no destruye el texto horizontal), sin shear/rotación agresiva, mosaic activo con `close_mosaic=20`

```bash
python training.py              # Entrena YOLO26s con dataset fusionado
python continue.py              # Resume desde checkpoint interrumpido
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
| **Mac Apple Silicon** | `mps` (auto) | `pip install ultralytics` y listo. Los env de ROCm se desactivan solos en Darwin. |
| **Linux AMD ROCm** | `0` (auto) | torch 2.4.1+rocm6.0 en Ubuntu tiene 2 bugs conocidos: `amdsmi` (NameError) y el crash de AdamW fused con AMP. Workarounds incluidos en script/entorno. |
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

## 🗺️ Roadmap / Mejores Mejoras Pendientes

- [ ] ⚠️ **SEGURIDAD**: `llm_config.json` contiene una API key real subida al repo — remover del tracking, ignorarla con .gitignore, y **rotar la key** (queda en el historial de commits)
- [ ] `continue.py` está stale: usa `data.yml` (dataset viejo) y settings AdamW de v8 — portarlo a YOLO26 + `data_merged.yml`
- [ ] `mine_negatives.py` hardcodea `HSA_OVERRIDE_GFX_VERSION` — extraer a un helper común de compatibilidad multiplataforma
- [ ] Mejorar `song_name`: anotar más ejemplos (39 de val es poco) o un split por caracteres con OCR
- [ ] Comparar `yolo26m` en Mac (entrenar 1+ día, mejor mAP si hay tiempo)
- [ ] Benchmark cuantizado: export int8/fp16 y medir pérdida de mAP vs latencia móvil

## 📜 Referencias

- [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26)
- [YOLOv8 vs YOLO26 Comparison](https://docs.ultralytics.com/compare/yolov8-vs-yolo26)
- Repo de inferencia offline: `piu_monolith`

---
*Desarrollado para la comunidad de Pump It Up.*
