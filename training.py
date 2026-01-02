from ultralytics import YOLO

def entrenar():
    # Cargamos el modelo YOLOv8 Nano (el mejor para apps de escritorio por su velocidad)
    model = YOLO('yolov8n.pt')

    # Entrenar
    model.train(
        data='data.yml',
        epochs=150,      # Si ves que el "loss" deja de bajar, puedes pararlo antes
        imgsz=1024,       # Resolución estándar
        device='mps',        # Usa la GPU si está disponible, si no, CPU
        batch=16,        # Ajusta según tu RAM
        project='piu_ia',
        name='first_try'
    )

if __name__ == '__main__':
    entrenar()