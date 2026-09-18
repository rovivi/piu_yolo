# PIU YOLO Training 💃🕹️

Este proyecto entrena un detector de objetos con **YOLO26** (Ultralytics) para reconocer elementos específicos de la pantalla del juego **Pump It Up (PIU)**.

## 🚀 Propósito
Identificar automáticamente información clave de la interfaz de PIU — nombre de la canción, puntaje, rango y dificultad — para sistemas de estadísticas automáticas, overlays o análisis de repeticiones.

## 📂 Estructura del Proyecto

- `training.py`: Script principal de entrenamiento (detecta el device automáticamente: MPS en Mac, ROCm/CUDA en Linux).
- `data.yml`: Dataset original (160 frames de video, split train/val).
- `data_merged.yml`: **Dataset fusionado** — frames de video + fotos de cabina (`photos/`). Este es el que usa el entrenamiento actual.
- `photos/`: Dataset de fotos de cabina / teléfono (58 imágenes, 44 train + 14 val).
- `mine_negatives.py`: Mining de falsos positivos → genera imágenes `neg_*` (fondo) para el train set.
- `prepare_dataset.py`: Split train/val del dataset.
- `continue.py`: Continuar entrenamiento desde el último checkpoint.
- `piu_ia/`: Runs de entrenamiento (checkpoints `best.pt`/`last.pt` + `args.yaml`). El run más reciente es `yolo26_v1`.
- `yolo26s.pt`: Base model YOLO26 Small.

## 🏷️ Clases Detectadas
1. `difficulty`: El nivel de dificultad de la canción.
2. `fullscore`: El puntaje máximo posible o acumulado.
3. `rank`: Los grados (S, SS, A, etc.).
4. `score`: El puntaje obtenido.
5. `song_name`: El título de la canción.

## 🛠️ Requisitos

```bash
pip install ultralytics   # >= 8.4.x (soporte YOLO26)
```

El modelo corre en cualquier plataforma:
- **Mac (Apple Silicon)**: usa MPS automáticamente.
- **Linux AMD (ROCm)**: solo en entornos con torch 2.4.1+rocm6.0 se requiere parchear `torch/cuda/__init__.py` (bug `amdsmi`) y pasar `amp=False` (ya incluidos en `training.py`).
- **Linux/CUDA** o CPU: funciona sin configuración extra.

## 🏋️ Entrenamiento

```bash
python training.py
```

### Configuración actual
- **Modelo**: YOLO26 Small (`yolo26s.pt`, NMS-free end-to-end, sin DFL)
- **Dataset**: `data_merged.yml` (204 train / 54 val, rutas relativas — portable Linux/Mac)
- **Épocas**: 300, patience 50, `imgsz=1024`, batch 16
- **Au mentaciones**: degrees 5°, sin fliplr/shear (song_name es texto horizontal)

### Resultados (val, dataset fusionado)

| Run | Modelo | mAP50 | mAP50-95 |
|---|---|---|---|
| `v5_photos` | YOLOv8 (cadena finetune) | 0.811 | 0.424 |
| `yolo26_v1` ✅ actual | YOLO26s scratch | **0.826** | **0.425** |

La clase `score` es la más sólida (mAP50 0.938); `song_name` es la más débil (0.614, texto largo, pocas imágenes).

## 📦 Convertir el modelo para app (Android/iOS)

```python
from ultralytics import YOLO
YOLO("piu_ia/yolo26_v1/weights/best.pt").export(format="ncnn")   # Android
YOLO("piu_ia/yolo26_v1/weights/best.pt").export(format="coreml") # iOS
```

---
*Desarrollado para la comunidad de Pump It Up.*
