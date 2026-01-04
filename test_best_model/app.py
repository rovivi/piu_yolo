import os
import cv2
import time
import threading
import datetime
import json
import numpy as np

# --- Configuración de Entorno (Optimización para AMD/ROCm) ---
# IMPORTANTE: Estas variables deben setearse ANTES de importar torch.
# Para GPUs AMD (ej. 9750XT/7900XTX), forzamos una arquitectura compatible.
if not os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    # AMD 9750XT suele ser gfx1031. Forzamos 10.3.0 para compatibilidad universal.
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Algunas versiones de ROCm requieren HCC_AMDGPU_TARGET para compilar kernels al vuelo
if not os.environ.get("HCC_AMDGPU_TARGET"):
    os.environ["HCC_AMDGPU_TARGET"] = "gfx1030"

# Desactivar SDMA si hay problemas de coherencia de memoria o timeouts en kernels de Linux
if not os.environ.get("HSA_ENABLE_SDMA"):
    os.environ["HSA_ENABLE_SDMA"] = "0"

# Verificación de pre-carga
print(f"[DEBUG] HSA_OVERRIDE_GFX_VERSION set to: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")

import torch
from PIL import Image, ImageTk
from ultralytics import YOLO
from tkinter import filedialog, messagebox

# Intentar importar Drag & Drop, fallback si falla
try:
    from tkinterdnd2 import TkinterDnD, DND_ALL
    HAS_DND = True
except ImportError:
    print("Aviso: tkinterdnd2 no instalado. Drag & Drop desactivado.")
    # Clase dummy para evitar errores si no está instalado
    class TkinterDnD:
        class DnDWrapper: pass
    HAS_DND = False

import customtkinter as ctk

# Tema Visual
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# --- Colores y Estilos ---
COLOR_ACCENT = "#3b8ed0"
COLOR_BG_CARD = "#2b2b2b"
COLOR_SUCCESS = "#2cc985"
COLOR_WARNING = "#e2b340"

class ImageItem:
    """Clase para gestionar el estado de cada imagen individualmente"""
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.processed = False
        self.detections = []  # Lista de dicts
        self.inference_time = 0.0
        self.image_tk = None  # Thumbnail
        self.result_image_cv = None # Imagen procesada en memoria (OpenCV)

class PIUAnalyticsPro(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        
        # Inicializar DND
        if HAS_DND:
            self.TkdndVersion = TkinterDnD._require(self)

        self.title("PIU Analytics Workstation | YOLOv8 Pro")
        self.geometry("1400x900")
        
        # Estado de la App
        self.model = None
        self.queue = [] # Lista de objetos ImageItem
        self.current_idx = -1
        self.device = "cpu"
        self.processing = False
        
        # Video State
        self.video_playing = False
        self.seek_to_frame = -1
        self.total_frames = 0
        self.fps_video = 30
        
        self.init_ui()
        self.init_backend()

    def init_ui(self):
        """Construye la interfaz compleja"""
        self.grid_columnconfigure(1, weight=1) # Panel central expandible
        self.grid_rowconfigure(1, weight=1)    # Panel central expandible

        # --- HEADER (Barra Superior) ---
        self.header = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color="#1a1a1a")
        self.header.grid(row=0, column=0, columnspan=3, sticky="ew")
        
        lbl_title = ctk.CTkLabel(self.header, text="PIU VISION ANALYTICS", font=("Roboto", 20, "bold"), text_color="white")
        lbl_title.pack(side="left", padx=20, pady=10)

        self.lbl_device = ctk.CTkLabel(self.header, text="INIT...", font=("Roboto Mono", 12), text_color="gray")
        self.lbl_device.pack(side="right", padx=20)

        # --- SIDEBAR (Izquierda - Controles) ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=1, column=0, sticky="nsew", rowspan=2)
        
        # Sección Modelo
        lbl_sets = ctk.CTkLabel(self.sidebar, text="CONFIGURACIÓN DEL MODELO", font=("Roboto", 12, "bold"))
        lbl_sets.pack(pady=(20, 10), padx=10, anchor="w")
        
        self.slider_conf = self.create_slider("Confianza Min.", 0.4)
        self.slider_iou = self.create_slider("IoU Threshold", 0.7)
        
        # Slider para Frame Skip (0 a 10) - Enteros
        frame_skip_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame_skip_frame.pack(fill="x", padx=20, pady=5)
        
        self.lbl_skip = ctk.CTkLabel(frame_skip_frame, text="Frame Skip: 0", anchor="w")
        self.lbl_skip.pack(fill="x")
        
        self.slider_skip = ctk.CTkSlider(frame_skip_frame, from_=0, to=10, number_of_steps=10,
                                         command=lambda v: self.lbl_skip.configure(text=f"Frame Skip: {int(v)}"))
        self.slider_skip.set(0)
        self.slider_skip.pack(fill="x", pady=5)

        # Botones de Acción
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").pack(fill="x", padx=10, pady=20)
        
        self.btn_import = ctk.CTkButton(self.sidebar, text="Importar Imágenes / Carpeta", command=self.import_files, height=40, fg_color="#4a4a4a")
        self.btn_import.pack(padx=20, pady=5, fill="x")

        self.btn_video = ctk.CTkButton(self.sidebar, text="Video en Tiempo Real", command=self.import_video, height=40, fg_color="#4a4a4a")
        self.btn_video.pack(padx=20, pady=(5, 10), fill="x")

        self.btn_run_all = ctk.CTkButton(self.sidebar, text="ANALIZAR COLA COMPLETA", command=self.start_batch_processing, height=50, fg_color=COLOR_ACCENT, font=("Roboto", 14, "bold"))
        self.btn_run_all.pack(padx=20, pady=10, fill="x")
        
        # Barra de Progreso
        self.progress_bar = ctk.CTkProgressBar(self.sidebar)
        self.progress_bar.set(0)
        self.progress_bar.pack(padx=20, pady=(10, 5), fill="x")
        self.lbl_progress = ctk.CTkLabel(self.sidebar, text="Listo", font=("Roboto", 11))
        self.lbl_progress.pack(padx=20, pady=0)

        # --- MAIN VIEW (Centro - Visor) ---
        self.main_view = ctk.CTkFrame(self, fg_color="#101010")
        self.main_view.grid(row=1, column=1, sticky="nsew")
        
        # Area de Drop
        self.canvas_area = ctk.CTkLabel(self.main_view, text="ARRASTRA IMÁGENES AQUÍ\n\n(O usa el botón importar)", font=("Roboto", 20), text_color="gray40")
        self.canvas_area.place(relx=0.5, rely=0.5, anchor="center")
        
        # Visor de Imagen Real (oculto al inicio)
        self.image_label = ctk.CTkLabel(self.main_view, text="")

        # --- CONTROLES DE VIDEO (Seekbar) ---
        self.video_controls = ctk.CTkFrame(self.main_view, fg_color="transparent")
        self.video_controls.pack(side="bottom", fill="x", padx=20, pady=10)
        
        self.seek_slider = ctk.CTkSlider(self.video_controls, from_=0, to=100, command=self.on_seek)
        self.seek_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.seek_slider.set(0)
        
        self.lbl_time = ctk.CTkLabel(self.video_controls, text="00:00 / 00:00", font=("Roboto Mono", 11))
        self.lbl_time.pack(side="right")
        self.video_controls.pack_forget() # Oculto hasta que haya video
        
        # Habilitar Drop en el área principal
        if HAS_DND:
            self.main_view.drop_target_register(DND_ALL)
            self.main_view.dnd_bind('<<Drop>>', self.on_drop)

        # --- STATS PANEL (Derecha - Datos) ---
        self.stats_panel = ctk.CTkScrollableFrame(self, width=300, label_text="ANÁLISIS EN VIVO")
        self.stats_panel.grid(row=1, column=2, sticky="nsew")
        
        self.txt_console = ctk.CTkTextbox(self.stats_panel, height=200, font=("Roboto Mono", 11))
        self.txt_console.pack(fill="x", padx=10, pady=10)
        self.txt_console.insert("0.0", "System Ready.\n")

        # Contenedor de estadísticas dinámicas
        self.stats_container = ctk.CTkFrame(self.stats_panel, fg_color="transparent")
        self.stats_container.pack(fill="both", expand=True, padx=10, pady=10)

        # --- FILMSTRIP (Inferior - Galería) ---
        self.filmstrip = ctk.CTkScrollableFrame(self, height=130, orientation="horizontal", label_text="Cola de Procesamiento")
        self.filmstrip.grid(row=2, column=1, columnspan=2, sticky="ew")

    def create_slider(self, text, default):
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=5)
        
        label = ctk.CTkLabel(frame, text=f"{text}: {default:.2f}", anchor="w")
        label.pack(fill="x")
        
        slider = ctk.CTkSlider(frame, from_=0, to=1, command=lambda v, l=label, t=text: l.configure(text=f"{t}: {v:.2f}"))
        slider.set(default)
        slider.pack(fill="x", pady=5)
        return slider

    def init_backend(self):
        # Hardware Check
        self.device = "cpu"
        self.log("--- Iniciando Diagnóstico de Hardware ---")
        
        try:
            # Info básica
            self.log(f"PyTorch Version: {torch.__version__}")
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            
            if is_rocm:
                self.log(f"Build: ROCm/HIP Detected (Version: {torch.version.hip})")
            else:
                self.log(f"Build: standard/CUDA Build (CUDA: {getattr(torch.version, 'cuda', 'N/A')})")

            # Intentar ver si ROCm detecta el hardware a pesar del available=False
            device_count = torch.cuda.device_count()
            self.log(f"GPUs detectadas por PyTorch: {device_count}")

            if torch.cuda.is_available():
                self.device = "cuda"
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, 'gcn_arch_name', 'N/A')
                
                if is_rocm:
                    status_text = f"GPU: {name} ({arch}) | ROCm {torch.version.hip}"
                    self.lbl_device.configure(text=status_text, text_color=COLOR_SUCCESS)
                    self.log(f"AMD GPU OPERATIVA: {name} ({arch})")
                else:
                    status_text = f"GPU: {name} | CUDA {torch.version.cuda}"
                    self.lbl_device.configure(text=status_text, text_color=COLOR_SUCCESS)
                    self.log(f"NVIDIA GPU OPERATIVA: {name}")
            else:
                self.lbl_device.configure(text="MODO CPU (No GPU detectada)", text_color=COLOR_WARNING)
                self.log("ALERTA: torch.cuda.is_available() es FALSE")
                if is_rocm:
                    self.log("Tip: Verifica que los drivers ROCm estén cargados y el usuario esté en el grupo 'render'/'video'")
                else:
                    self.log("Tip: Estás usando un build de PyTorch sin soporte para AMD. Reinstala con soporte ROCm.")
                
            # Log de overrides activos
            self.log(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'None')}")
            self.log(f"HSA_ENABLE_SDMA: {os.environ.get('HSA_ENABLE_SDMA', 'None')}")
            self.log("-----------------------------------------")
            
        except Exception as e:
            self.log(f"Error crítico en init_backend: {e}")
            self.lbl_device.configure(text="Hardware Error", text_color="#ff4444")

        # Cargar Modelo
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _load_model_thread(self):
        self.log("Cargando motor YOLOv8...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Rutas de búsqueda
        paths = [
            os.path.join(base_dir, 'best.pt'),
            os.path.join(os.path.dirname(base_dir), 'piu_ia/first_try4/weights/best.pt')
        ]
        
        model_path = None
        for p in paths:
            if os.path.exists(p):
                model_path = p
                break
        
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.log(f"Modelo cargado: {os.path.basename(model_path)}")
            except Exception as e:
                self.log(f"Error crítico: {e}")
        else:
            self.log("ERROR: No se encontró best.pt")

    def log(self, msg):
        self.txt_console.configure(state="normal")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_console.insert("end", f"[{ts}] {msg}\n")
        self.txt_console.see("end")
        self.txt_console.configure(state="disabled")

    # --- Gestión de Archivos ---
    def on_drop(self, event):
        files = self.split_list(event.data)
        self.add_images(files)

    def split_list(self, data):
        # TkinterDnD devuelve strings raros con {} a veces
        if data.startswith('{') and data.endswith('}'):
            # Manejo básico de rutas con espacios en Linux
            return [data[1:-1]] 
        return data.split()

    def import_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Imágenes", "*.jpg *.png *.jpeg *.webp")])
        if files:
            self.add_images(list(files))

    def add_images(self, paths):
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        added_count = 0
        
        for p in paths:
            # Limpieza básica de rutas (a veces el dnd mete caracteres extra)
            p = p.strip('{}') 
            if p.lower().endswith(valid_exts) and os.path.exists(p):
                item = ImageItem(p)
                self.queue.append(item)
                self.create_thumbnail(item, len(self.queue)-1)
                added_count += 1
        
        if added_count > 0:
            self.log(f"Agregadas {added_count} imágenes a la cola.")
            if self.current_idx == -1:
                self.select_image(0)
            self.canvas_area.place_forget() # Ocultar placeholder

    def create_thumbnail(self, item, idx):
        # Crear botón pequeño en el filmstrip
        try:
            img = Image.open(item.path)
            img.thumbnail((100, 100))
            ctk_img = ctk.CTkImage(img, size=(80, 60))
            
            btn = ctk.CTkButton(self.filmstrip, image=ctk_img, text=f"{idx+1}", width=90, compound="top",
                                fg_color="transparent", border_width=1, border_color="gray",
                                command=lambda i=idx: self.select_image(i))
            btn.pack(side="left", padx=5, pady=5)
            item.ui_btn = btn # Guardar referencia para cambiar color luego
        except Exception as e:
            print(f"Error thumbnail: {e}")

    def select_image(self, idx):
        if idx < 0 or idx >= len(self.queue): return
        
        self.current_idx = idx
        item = self.queue[idx]
        
        # Mostrar en visor central
        self.display_main_image(item)
        
        # Actualizar Stats si ya fue procesada
        if item.processed:
            self.update_stats_panel(item)
        else:
            self.clear_stats_panel()

    def display_main_image(self, item):
        # Elegir fuente (Original o Procesada)
        if item.result_image_cv is not None:
            # Convertir BGR a RGB
            color_img = cv2.cvtColor(item.result_image_cv, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(color_img)
        else:
            pil_img = Image.open(item.path)
            
        # Redimensionar dinámicamente al tamaño del visor
        w_avail = self.main_view.winfo_width()
        h_avail = self.main_view.winfo_height()
        if w_avail < 200: w_avail = 800
        if h_avail < 200: h_avail = 600
        
        # Calcular ratio
        ratio = min(w_avail / pil_img.width, h_avail / pil_img.height)
        new_w = int(pil_img.width * ratio * 0.95) # 95% para margen
        new_h = int(pil_img.height * ratio * 0.95)
        
        ctk_img = ctk.CTkImage(pil_img, size=(new_w, new_h))
        self.image_label.configure(image=ctk_img)
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

    # --- Motor de Procesamiento ---
    def start_batch_processing(self):
        if not self.model: return
        if self.processing: return
        
        self.processing = True
        self.btn_run_all.configure(state="disabled", text="PROCESANDO...")
        
        # Hilo separado para no congelar GUI
        threading.Thread(target=self._process_queue, daemon=True).start()

    def _process_queue(self):
        total = len(self.queue)
        conf = self.slider_conf.get()
        iou = self.slider_iou.get()
        
        for i, item in enumerate(self.queue):
            if not item.processed:
                start_t = time.time()
                
                # 1. Leer Imagen
                img0 = cv2.imread(item.path)
                
                # 2. Inferencia
                results = self.model.predict(img0, conf=conf, iou=iou, verbose=False)
                res = results[0]
                
                # 3. Guardar Datos
                item.inference_time = (time.time() - start_t) * 1000
                item.result_image_cv = res.plot() # Bounding boxes pintadas
                
                item.detections = []
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    item.detections.append({
                        "class": self.model.names[cls_id],
                        "conf": float(box.conf[0])
                    })
                
                item.processed = True
                
                # 4. Actualizar UI (Thread safe call)
                self.after(0, lambda idx=i: self._update_progress_ui(idx, total))
        
        self.after(0, self._finish_processing)

    def _update_progress_ui(self, idx, total):
        progress = (idx + 1) / total
        self.progress_bar.set(progress)
        self.lbl_progress.configure(text=f"Procesando: {idx+1}/{total}")
        
        # Marcar thumbnail como verde
        item = self.queue[idx]
        if hasattr(item, 'ui_btn'):
            item.ui_btn.configure(border_color=COLOR_SUCCESS, border_width=2)
            
        # Si es la imagen que estamos viendo, actualizar visor
        if idx == self.current_idx:
            self.select_image(idx)

    def _finish_processing(self):
        self.processing = False
        self.btn_run_all.configure(state="normal", text="ANALIZAR COLA COMPLETA")
        self.lbl_progress.configure(text="Proceso Completado")
        self.log(f"Lote finalizado. Total imágenes: {len(self.queue)}")
        messagebox.showinfo("Éxito", "Análisis completado.")

    # --- Panel de Estadísticas ---
    def clear_stats_panel(self):
        for widget in self.stats_container.winfo_children():
            widget.destroy()

    def update_stats_panel(self, item):
        self.clear_stats_panel()
        
        # Titulo
        ctk.CTkLabel(self.stats_container, text="Resumen de Detección", font=("Roboto", 14, "bold")).pack(anchor="w", pady=5)
        
        # Tiempo
        ctk.CTkLabel(self.stats_container, text=f"Tiempo: {item.inference_time:.1f}ms", text_color=COLOR_ACCENT).pack(anchor="w")
        
        # Conteo de Clases
        counts = {}
        for d in item.detections:
            name = d['class']
            counts[name] = counts.get(name, 0) + 1
            
        if not counts:
            ctk.CTkLabel(self.stats_container, text="Sin detecciones", text_color="gray").pack(pady=10)
        else:
            # Generar barras simples
            for name, count in counts.items():
                frame = ctk.CTkFrame(self.stats_container, fg_color="transparent")
                frame.pack(fill="x", pady=2)
                
                ctk.CTkLabel(frame, text=f"{name} ({count})", width=100, anchor="w").pack(side="left")
                
                # Barra visual
                bar_width = min(count * 20, 150) # Escala visual
                bar = ctk.CTkFrame(frame, width=bar_width, height=10, fg_color=COLOR_SUCCESS)
                bar.pack(side="left", padx=5)

    # --- Video Real-time ---
    def import_video(self):
        file = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if file:
            self.start_video_processing(file)

    def start_video_processing(self, video_path):
        if self.processing: return
        self.stop_video_processing()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log("Error al abrir video")
            return
            
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_video = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        self.processing = True
        self.video_playing = True
        self.seek_to_frame = -1
        
        # UI updates
        self.btn_run_all.configure(state="disabled")
        self.btn_video.configure(text="DETENER VIDEO", fg_color=COLOR_WARNING, command=self.stop_video_processing)
        self.video_controls.pack(side="bottom", fill="x", padx=20, pady=10)
        self.seek_slider.configure(to=self.total_frames)
        self.seek_slider.set(0)
        self.canvas_area.place_forget()
        
        self.log(f"Iniciando video: {os.path.basename(video_path)} ({self.total_frames} frames)")
        
        threading.Thread(target=self._process_video_thread, args=(video_path,), daemon=True).start()

    def stop_video_processing(self):
        self.video_playing = False
        
    def on_seek(self, value):
        self.seek_to_frame = int(value)

    def _process_video_thread(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.log("Error en hilo de video")
            self.after(0, self._finish_video)
            return

        frame_count = 0
        last_res_plotted = None
        
        while self.video_playing and cap.isOpened():
            # Handle seeking
            if self.seek_to_frame != -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_to_frame)
                frame_count = self.seek_to_frame
                self.seek_to_frame = -1

            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            skip_rate = int(self.slider_skip.get())
            should_run = (skip_rate == 0) or (frame_count % (skip_rate + 1) == 0)
            
            start_t = time.time()
            
            if should_run or last_res_plotted is None:
                conf = self.slider_conf.get()
                iou = self.slider_iou.get()
                results = self.model.predict(frame, conf=conf, iou=iou, verbose=False)
                res = results[0]
                last_res_plotted = res.plot()
                cv2.putText(last_res_plotted, "INF", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                display_img = frame.copy()
                cv2.putText(display_img, ".", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                last_res_plotted = display_img # Reuse for consistency? actually display_img is what we want

            display_img = last_res_plotted
            
            # FPS & Info
            proc_time = time.time() - start_t
            fps = 1.0 / (proc_time + 1e-6)
            cv2.putText(display_img, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update UI
            color_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(color_img)
            
            # Update slider and frame
            self.after(0, lambda img=pil_img, cur=frame_count: self._update_video_frame(img, cur))

        cap.release()
        self.after(0, self._finish_video)

    def _update_video_frame(self, pil_img, current_frame):
        if not self.video_playing: return
        
        # Update Seekbar without triggering command if possible 
        # (Standard CTkSlider doesn't have a silent set, but we handle it by flag if needed)
        # For simple seek, we just set it.
        self.seek_slider.set(current_frame)
        
        # Update Time Label
        cur_sec = int(current_frame / self.fps_video)
        tot_sec = int(self.total_frames / self.fps_video)
        self.lbl_time.configure(text=f"{cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}")

        w_avail = self.main_view.winfo_width()
        h_avail = self.main_view.winfo_height() - 60 # Margen para controles
        if w_avail < 10 or h_avail < 10: return
        
        ratio = min(w_avail / pil_img.width, h_avail / pil_img.height)
        new_w = int(pil_img.width * ratio)
        new_h = int(pil_img.height * ratio)
        
        ctk_img = ctk.CTkImage(pil_img, size=(new_w, new_h))
        self.image_label.configure(image=ctk_img)
        self.image_label.place(relx=0.5, rely=0.45, anchor="center")

    def _finish_video(self):
        self.processing = False
        self.video_playing = False
        self.btn_run_all.configure(state="normal")
        self.btn_video.configure(text="Video en Tiempo Real", fg_color="#4a4a4a", command=self.import_video)
        self.video_controls.pack_forget()
        self.log("Video finalizado.")

if __name__ == "__main__":
    app = PIUAnalyticsPro()
    app.mainloop()