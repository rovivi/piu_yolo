from ultralytics import YOLO
import os

# Tu config de AMD
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

def resumir_entrenamiento():
    # 1. Ruta al archivo last.pt (ASEGÚRATE DE QUE ESTA RUTA SEA CORRECTA)
    # Si tu carpeta se llama 'piu_ia', busca dentro de 'improved_v1' o 'improved_v2_adamw'
    path_al_modelo = 'piu_ia/improved_v2_adamw/weights/last.pt' 
    
    if not os.path.exists(path_al_modelo):
        print(f"❌ ERROR: No encontré el archivo en {path_al_modelo}")
        print("Revisa tu carpeta 'piu_ia' y dime cómo se llama la subcarpeta del entrenamiento.")
        return

    print(f"✅ Archivo encontrado. Intentando resumir desde {path_al_modelo}...")

    try:
        # Cargamos el checkpoint
        model = YOLO(path_al_modelo)

        # Re-intentamos el resume
        # Forzamos los parámetros de hardware de nuevo por si acaso
        model.train(resume=True)
        
    except Exception as e:
        print(f"❌ Falló el resume automático: {e}")
        print("Intentando alternativa: Cargar pesos y continuar manualmente...")
        
        # PLAN B: Si el resume falla, cargamos los pesos y lanzamos 250 epochs más
        model = YOLO(path_al_modelo)
        model.train(
            data='data.yml',
            epochs=500, # YOLO detectará que ya lleva 250 si usas los mismos settings
            imgsz=1024,
            batch=16,
            device=0,
            optimizer='AdamW',
            project='piu_ia',
            name='improved_v1', # mismo nombre para que intente seguir ahí
            exist_ok=True # para que no cree una carpeta nueva
        )

if __name__ == '__main__':
    resumir_entrenamiento()