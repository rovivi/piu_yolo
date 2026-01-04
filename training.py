from ultralytics import YOLO
import os

# Tu configuración para AMD/ROCm
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

def entrenar():
    # Cargamos el modelo
    model = YOLO('yolov8n.pt')

    # Entrenamos
    model.train(
        data='data.yml',
        
        # --- DURACIÓN ---
        # Aumentamos epochs. Al ser un modelo nano, aprende rápido, 
        # pero necesita refinar mucho.
        epochs=500, 
        # Subimos la paciencia. A veces el modelo se estanca 30 epochs 
        # y luego da un salto de mejora.
        patience=50,      
        
        imgsz=1024,
        batch=16,
        device=0, 
        
        # --- OPTIMIZACIÓN DEL MODELO ---
        # 'AdamW' suele converger mejor y más rápido en datasets pequeños/custom 
        # que el SGD por defecto.
        optimizer='AdamW',
        # Un Learning Rate inicial un poco más bajo para no "romper" 
        # los pesos pre-entrenados demasiado rápido.
        lr0=0.001, 
        # Usamos Cosine LR para suavizar la bajada del learning rate
        cos_lr=True,

        # --- AUGMENTATION GEOMÉTRICO ---
        degrees=50.0,     # Bajé un poco de 15 a 10 (textos rotados son difíciles)
        translate=0.1,    
        scale=0.6,        # Aumenté un poco el rango de escala
        shear=1.0,        # OJO: Shear alto deforma mucho letras/números. Lo bajé a 0 o muy bajo.
        fliplr=0.0,       # IMPORTANTE: Si detectas TEXTO o NÚMEROS, pon esto en 0.0.
                          # Un "5" al revés no existe. Si detectas objetos simétricos, déjalo en 0.5.
        mosaic=1.0,       
        mixup=0.15,       # Subí un poco el mixup
        
        # --- AUGMENTATION DE COLOR (Vital para pantallas/arcades) ---
        # Simula cambios de tono (hue), saturación y brillo (value)
        hsv_h=0.015, 
        hsv_s=0.7,        # Alta variación de saturación (luces neón)
        hsv_v=0.4,        # Alta variación de brillo (pantalla brillante vs oscura)

        # --- ESTRATEGIA FINAL ---
        # Desactiva el efecto Mosaic las últimas 20 epochs.
        # Esto hace que al final el modelo entrene con imágenes "reales" y no recortadas,
        # mejorando mucho la precisión final.
        close_mosaic=20,

        # Regularización
        weight_decay=0.0005, 
        
        project='piu_ia',
        name='improved_v2_adamw' # Nombre nuevo para no sobreescribir
    )

if __name__ == '__main__':
    entrenar()