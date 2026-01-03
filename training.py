from ultralytics import YOLO
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
def entrenar():
    # Cargamos el modelo YOLOv8 Nano
    model = YOLO('yolov8n.pt')

    # Configuración avanzada para mejorar el dataset pequeño
    model.train(
        data='data.yml',
        epochs=150,
        imgsz=1024,
        
        # --- HARDWARE ---
        # Para AMD (usando ROCm en Linux o DirectML en Windows), 
        # generalmente se usa device=0 o device='0'.
        device=0, 
        batch=16,
        
        # --- MEJORAS PARA DATASET PEQUEÑO (Data Augmentation) ---
        # Rotamos las imágenes +/- 15 grados
        degrees=15.0,     
        # Desplazamiento horizontal/vertical (0.1 = 10%)
        translate=0.1,    
        # Escala aleatoria (0.5 = puede ser 50% más pequeña o grande)
        scale=0.5,        
        # Deformación de perspectiva
        shear=2.0,        
        # Volteo horizontal (50% de probabilidad)
        fliplr=0.5,       
        # Volteo vertical (útil si tus fotos no tienen un "arriba" fijo)
        flipud=0.0,       
        # Mosaic: Combina 4 imágenes en una (Esencial en YOLO)
        mosaic=1.0,       
        # Mixup: Mezcla dos imágenes (Muy bueno para evitar overfitting)
        mixup=0.1,        
        
        # --- CONTROL DE OVERFITTING ---
        # Early Stopping: Si en 20 epochs no mejora el mAP50, se detiene
        patience=25,      
        # Regularización: Ayuda a que los pesos no se vuelvan locos
        weight_decay=0.0005, 
        
        # --- PROYECTO ---
        project='piu_ia',
        name='improved_v1'
    )

if __name__ == '__main__':
    entrenar()