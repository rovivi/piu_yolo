# PIU YOLO Training 💃🕹️

Este proyecto está diseñado para entrenar un modelo de detección de objetos utilizando **YOLOv8** para reconocer elementos específicos en la pantalla del juego **Pump It Up (PIU)**.

## 🚀 Propósito
El objetivo principal es identificar automáticamente información clave de la interfaz de usuario de PIU, como el nombre de la canción, el puntaje, el rango y la dificultad. Esto puede ser útil para sistemas de estadísticas automáticas, overlays o análisis de repeticiones.

## 📂 Estructura del Proyecto

- `training.py`: Script principal para iniciar el entrenamiento del modelo.
- `data.yml`: Configuración del dataset (rutas de imágenes y nombres de clases).
- `classes.txt`: Lista de las etiquetas/clases que el modelo aprenderá a detectar.
- `images/`: Directorio que contiene las capturas de pantalla para el entrenamiento.
- `labels/`: Directorio con las anotaciones en formato YOLO para cada imagen.
- `yolov8n.pt`: Pesos iniciales del modelo YOLOv8 Nano (modelo ligero y rápido).

## 🏷️ Clases Detectadas
El modelo está configurado para reconocer las siguientes 5 clases:
1. `difficulty`: El nivel de dificultad de la canción.
2. `fullscore`: El puntaje máximo posible o acumulado.
3. `rank`: Los grados (S, SS, A, etc.).
4. `score`: El puntaje obtenido.
5. `song_name`: El título de la canción.

## 🛠️ Requisitos
Asegúrate de tener instalada la librería de Ultralytics:

```bash
pip install ultralytics
```

## 🏋️ Entrenamiento
Para comenzar el entrenamiento, simplemente ejecuta el script `training.py`:

```bash
python training.py
```

### Configuración de Entrenamiento
El script está configurado actualmente con:
- **Modelo**: YOLOv8 Nano (`yolov8n.pt`)
- **Épocas**: 100
- **Resolución**: 1024px
- **Dispositivo**: `mps` (optimizado para chips Apple Silicon) o detectará automáticamente tu hardware.
- **Proyecto**: Los resultados se guardarán en la carpeta `piu_ia/first_try`.

---
*Desarrollado para la comunidad de Pump It Up.*
