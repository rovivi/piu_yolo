import os
import sys
import threading
import shutil
import subprocess
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Dict, Optional
import warnings

# Filter specific torchvision warning about executable stack
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2

# --- Configuración de Entorno (Optimización para AMD/ROCm & Apple M2) ---
# IMPORTANTE: Estas variables deben setearse ANTES de importar torch.
if "linux" in sys.platform:
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
        
    # PyTorch ROCm optimizaciones
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"

# Apple Metal (MPS) optimizaciones - Solo si corre en Mac
if sys.platform == "darwin":
     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Import Core Logic
from piu_core import VideoProcessor, FrameAnalyzer, LLMService
import json

# --- Configuration & Constants ---
APP_NAME = "PIU Analytics Pro V2.5"
VERSION = "2.5.0"

# Modern Color Palette
COLOR_BG_DARK = "#121212"
COLOR_BG_CARD = "#1e1e1e"
COLOR_ACCENT = "#3498db"         # Soft Blue
COLOR_ACCENT_HOVER = "#2980b9"
COLOR_SUCCESS = "#27ae60"        # Green
COLOR_WARNING = "#f39c12"        # Orange
COLOR_ERROR = "#c0392b"          # Red

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class PIUApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title(f"{APP_NAME} v{VERSION}")
        self.geometry("1300x900")
        self.configure(fg_color=COLOR_BG_DARK)
        
        # --- Logic Components ---
        self.processor = VideoProcessor()
        self.analyzer: Optional[FrameAnalyzer] = None
        self.valid_frames = []
        self.is_processing = False
        
        # LLM State
        self.llm_service = None
        self.llm_config = {"api_url": "https://api.openai.com/v1/chat/completions", "api_key": "", "model": "gpt-4o"}
        self._load_config()
        
        # Load Model
        self._set_status("Initializing Model...", COLOR_WARNING)
        threading.Thread(target=self._load_model, daemon=True).start()
        
        self._init_ui()
        
    def _init_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#181818")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        self._build_sidebar()
        
        # --- Main Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Tabs
        self.video_tab = self._create_video_tab(self.main_frame)
        self.video_tab.grid(row=0, column=0, sticky="nsew")
        
    def _build_sidebar(self):
        lbl_logo = ctk.CTkLabel(self.sidebar, text="PIU ANALYTICS", font=("Montserrat", 22, "bold"), text_color=COLOR_ACCENT)
        lbl_logo.pack(pady=30, padx=20)
        
        # Settings
        ctk.CTkLabel(self.sidebar, text="SETTINGS", font=("Roboto", 12, "bold"), text_color="#555").pack(anchor="w", padx=25, pady=(20,5))
        
        self.combo_device = ctk.CTkComboBox(self.sidebar, values=["Auto (GPU/CPU)", "CPU Only"], fg_color="#2b2b2b", border_color="#444")
        self.combo_device.pack(pady=5, padx=20, fill="x")

        # -- AI Settings --
        ctk.CTkLabel(self.sidebar, text="AI SETTINGS", font=("Roboto", 12, "bold"), text_color="#555").pack(anchor="w", padx=25, pady=(20,5))
        
        self.entry_api_url = ctk.CTkEntry(self.sidebar, placeholder_text="API Endpoint", height=30, border_color="#444")
        self.entry_api_url.bind("<FocusOut>", lambda e: self._save_config_params()) # Auto-save/update state
        self.entry_api_url.pack(padx=20, pady=5, fill="x")
        self.entry_api_url.insert(0, self.llm_config.get("api_url", ""))
        
        self.entry_api_key = ctk.CTkEntry(self.sidebar, placeholder_text="API Key (sk-...)", height=30, border_color="#444", show="*")
        self.entry_api_key.bind("<FocusOut>", lambda e: self._save_config_params())
        self.entry_api_key.pack(padx=20, pady=5, fill="x")
        self.entry_api_key.insert(0, self.llm_config.get("api_key", ""))
        
        self.entry_model = ctk.CTkEntry(self.sidebar, placeholder_text="Model (gpt-4o)", height=30, border_color="#444")
        self.entry_model.bind("<FocusOut>", lambda e: self._save_config_params())
        self.entry_model.pack(padx=20, pady=5, fill="x")
        self.entry_model.insert(0, self.llm_config.get("model", ""))
        
        self.btn_init_ai = ctk.CTkButton(self.sidebar, text="Reload AI Service", command=self._init_llm, 
                                       fg_color="#5e35b1", hover_color="#4527a0", height=30)
        self.btn_init_ai.pack(padx=20, pady=(10, 5), fill="x") # Reduced bottom padding

        self.btn_list_models = ctk.CTkButton(self.sidebar, text="List Available Models", command=self._list_server_models, 
                                       fg_color="#333", hover_color="#444", height=24, font=("Roboto", 10))
        self.btn_list_models.pack(padx=20, pady=(0, 10), fill="x")
        
        self.btn_clean = ctk.CTkButton(self.sidebar, text="Clean Cache", command=self.clean_project_cache, 
                                     fg_color="#c0392b", hover_color="#a93226")
        self.btn_clean.pack(pady=20, padx=20, fill="x")

        # Adding new buttons as per instruction
        # self.btn_video = ctk.CTkButton(self.sidebar, text="Video en Tiempo Real", command=self.import_video, height=40, fg_color="#4a4a4a")
        # self.btn_video.pack(padx=20, pady=(5, 10), fill="x") # This button was not in the original code, but in the diff. I'll add it.

        self.btn_url = ctk.CTkButton(self.sidebar, text="Importar desde URL", command=self.import_url, height=40, fg_color="#4a4a4a")
        self.btn_url.pack(padx=20, pady=(5, 10), fill="x")
        
        # App Log / Detailed Status
        ctk.CTkLabel(self.sidebar, text="ACTIVITY LOG", font=("Roboto", 12, "bold"), text_color="#555").pack(anchor="w", padx=25, pady=(10,5))
        self.log_text = ctk.CTkTextbox(self.sidebar, fg_color="#101010", text_color="#aaa", font=("Roboto Mono", 10), height=300)
        self.log_text.pack(fill="x", padx=20, pady=5)
        
        # Footer
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Ready", font=("Roboto Mono", 11), text_color="gray", wraplength=240, justify="left")
        self.lbl_status.pack(side="bottom", pady=20, padx=20, anchor="w")

    def _create_video_tab(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1) # Results expand
        
        # 1. Input Section
        input_card = ctk.CTkFrame(frame, fg_color=COLOR_BG_CARD, corner_radius=10)
        input_card.grid(row=0, column=0, sticky="ew", padx=30, pady=20)
        
        ctk.CTkLabel(input_card, text="INPUT SOURCE", font=("Roboto", 14, "bold"), text_color=COLOR_ACCENT).pack(anchor="w", padx=20, pady=(15,5))
        
        self.entry_url = ctk.CTkEntry(input_card, placeholder_text="Paste YouTube URL here...", height=45, border_color="#444")
        self.entry_url.pack(fill="x", padx=20, pady=10)
        
        btn_box = ctk.CTkFrame(input_card, fg_color="transparent")
        btn_box.pack(fill="x", padx=20, pady=10)
        
        self.btn_download = ctk.CTkButton(btn_box, text="Download & Analyze", command=self.start_download_process, 
                                        fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, height=40)
        self.btn_download.pack(side="left", fill="x", expand=True, padx=(0,10))
        
        ctk.CTkLabel(btn_box, text="OR", text_color="gray").pack(side="left")
        
        self.btn_file = ctk.CTkButton(btn_box, text="Open Local File", command=self.select_local_video,
                                    fg_color="#444", hover_color="#555", height=40)
        self.btn_file.pack(side="left", fill="x", expand=True, padx=(10,0))
        
        # Side buttons for URL (Added feature)
        self.btn_url_imp = ctk.CTkButton(btn_box, text="Import URL", command=self.import_url, width=100, fg_color="#444")
        self.btn_url_imp.pack(side="left", padx=(10,0))


        # 2. Status & Progress
        status_card = ctk.CTkFrame(frame, fg_color=COLOR_BG_CARD, corner_radius=10)
        status_card.grid(row=1, column=0, sticky="ew", padx=30, pady=(0,20))
        
        self.meta_frame = ctk.CTkFrame(status_card, fg_color="transparent")
        self.lbl_meta_title = ctk.CTkLabel(self.meta_frame, text="Video Title", font=("Roboto", 14, "bold"))
        self.lbl_meta_title.pack(anchor="w")
        self.lbl_meta_info = ctk.CTkLabel(self.meta_frame, text="Duration: --:--", text_color="gray")
        self.lbl_meta_info.pack(anchor="w")
        
        self.lbl_progress_act = ctk.CTkLabel(status_card, text="Idle", font=("Roboto", 12))
        self.lbl_progress_act.pack(anchor="w", padx=20, pady=(15,5))
        
        self.progress_bar = ctk.CTkProgressBar(status_card, height=12, corner_radius=6)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=20, pady=(0,20))

        # 3. Gallery
        res_frame = ctk.CTkFrame(frame, fg_color="transparent")
        res_frame.grid(row=2, column=0, sticky="nsew", padx=30, pady=(0,30))
        
        header = ctk.CTkFrame(res_frame, fg_color="transparent")
        header.pack(fill="x", pady=(0,10))
        
        ctk.CTkLabel(header, text="DETECTED EVENTS (Strict Mode)", font=("Roboto", 16, "bold"), text_color="white").pack(side="left")
        
        self.btn_export = ctk.CTkButton(header, text="Export Results", state="disabled", fg_color=COLOR_SUCCESS, command=self.export_frames)
        self.btn_export.pack(side="right")

        self.scroll_results = ctk.CTkScrollableFrame(res_frame, fg_color="#181818", label_text="Click image to enlarge")
        self.scroll_results.pack(fill="both", expand=True)

        return frame

        return frame

    def _load_config(self):
        try:
            if os.path.exists("llm_config.json"):
                with open("llm_config.json", "r") as f:
                    self.llm_config.update(json.load(f))
        except Exception as e:
            print(f"Config load error: {e}")
            
    def _save_config_params(self):
        self.llm_config["api_url"] = self.entry_api_url.get().strip()
        self.llm_config["api_key"] = self.entry_api_key.get().strip()
        self.llm_config["model"] = self.entry_model.get().strip()
        
        try:
            with open("llm_config.json", "w") as f:
                json.dump(self.llm_config, f)
        except Exception as e:
            print(f"Config save error: {e}")

    def _init_llm(self):
        self._save_config_params()
        if not self.llm_config["api_key"]:
            self._log("AI Init: No API Key provided.")
            return

        try:
            self.llm_service = LLMService(
                self.llm_config["api_url"],
                self.llm_config["api_key"],
                self.llm_config["model"]
            )
            
            # Auto-detect model if missing
            if not self.llm_config["model"]:
                self._log("Model not set. Auto-detecting...")
                threading.Thread(target=self._auto_select_model, daemon=True).start()
            else:
                self._log("AI Service initialized.")
                self._set_status(f"AI Ready: {self.llm_config['model']}", COLOR_SUCCESS)
                
        except Exception as e:
            self._log(f"AI Init Error: {e}")

    def _auto_select_model(self):
        try:
            models = self.llm_service.list_available_models()
            if models:
                first = models[0]
                self.llm_config["model"] = first
                self.llm_service.model = first
                
                # UI Update in main thread
                self.after(0, lambda: [
                    self.entry_model.delete(0, "end"),
                    self.entry_model.insert(0, first),
                    self._save_config_params(),
                    self._set_status(f"AI Ready (Auto): {first}", COLOR_SUCCESS),
                    self._log(f"Auto-selected model: {first}")
                ])
            else:
                self.after(0, lambda: self._log("Could not auto-detect models."))
        except Exception as e:
             self.after(0, lambda: self._log(f"Auto-detect error: {e}"))

    def _list_server_models(self):
        if not self.llm_service:
            self._log("AI Service not initialized.")
            return
            
        def thread_list():
            self._log("Fetching models...")
            models = self.llm_service.list_available_models()
            if models:
                msg = "\n".join([f"- {m}" for m in models])
                self._log(f"Available Models:\n{msg}")
                messagebox.showinfo("Models", f"Found {len(models)} models. Check Log for details.")
            else:
                self._log("No models found or error fetching.")
                messagebox.showwarning("Models", "No models found.")
        
        threading.Thread(target=thread_list, daemon=True).start()

    # --- Logic ---
    def _load_model(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, 'best.pt'),
            os.path.join(os.path.dirname(base_dir), 'piu_ia/first_try4/weights/best.pt')
        ]
        model_path = None
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break
        
        if model_path:
            try:
                self.analyzer = FrameAnalyzer(model_path)
                self._set_status(f"Model Active: {os.path.basename(model_path)}", COLOR_SUCCESS)
                
                # Show GPU Info
                gpu_info = getattr(self.analyzer, 'gpu_name', 'Unknown Device')
                self._log(f"Model loaded: {model_path}")
                self._log(f"HARDWARE: {gpu_info}")
                self.combo_device.set(gpu_info)
                
            except Exception as e:
                self._set_status(f"Model Init Error: {e}", COLOR_ERROR)
        else:
            self._set_status("Error: 'best.pt' not found.", COLOR_ERROR)

    def select_local_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")])
        if path:
            self.start_processing_pipeline(path)

    def import_url(self):
        # Dialogo simple para URL
        dialog = ctk.CTkInputDialog(text="Introduce la URL del video:", title="Importar URL")
        url = dialog.get_input()
        if url:
             if not (url.startswith("http://") or url.startswith("https://")):
                 messagebox.showerror("Error", "URL Inválida. Debe iniciar con http/https")
                 return
             
             self._log(f"URL recibida: {url}")
             # Run download pipeline
             self._set_ui_state(processing=True)
             self.meta_frame.pack_forget()
             threading.Thread(target=self._pipeline_download, args=(url,), daemon=True).start()

    def start_download_process(self):
        url = self.entry_url.get()
        if not url:
            messagebox.showwarning("Input Error", "Enter a URL or select a local file.")
            return
            
        self._set_ui_state(processing=True)
        self.meta_frame.pack_forget()
        
        threading.Thread(target=self._pipeline_download, args=(url,), daemon=True).start()
        
    def _pipeline_download(self, url):
        try:
            self._log(f"Starting download: {url}")
            self._update_progress(0, "Connecting to YouTube...")
            
            def hook(d):
                if d['status'] == 'downloading':
                    try:
                        p_str = d.get('_percent_str', '0%')
                        p = float(p_str.replace('%','')) / 100
                        speed = d.get('_speed_str', 'N/A')
                        eta = d.get('_eta_str', 'N/A')
                        
                        msg = f"Downloading: {p_str} | Speed: {speed} | ETA: {eta}"
                        self._update_progress(p, msg)
                    except: pass
            
            vid_data = self.processor.download_url(url, progress_hook=hook)
            self._log("Download complete.")
            
            self.after(0, lambda: self._show_meta(vid_data))
            self.start_processing_pipeline(vid_data["path"], threaded=False)
            
        except Exception as e:
            self._handle_error(e)

    def start_processing_pipeline(self, vid_path, threaded=True):
        # Cancel previous if running
        if self.is_processing:
            self.is_processing = False

        if threaded:
            self._set_ui_state(processing=True)
            threading.Thread(target=self._process_video_logic, args=(vid_path,), daemon=True).start()
        else:
            self._process_video_logic(vid_path)
            
    def _process_video_logic(self, vid_path):
        try:
            self._log(f"Beginning processing for: {os.path.basename(vid_path)}")
            self.is_processing = True

            def ffmpeg_cb(curr, total, msg):
                if not self.is_processing: return
                self._update_progress(curr, msg)
                
            # 1. Extract Frames
            self._log("Extracting frames (using all CPU cores)...")
            frames = self.processor.extract_frames(vid_path, progress_callback=ffmpeg_cb)
            if not self.is_processing: 
                self._log("Processing cancelled during frame extraction.")
                self.after(0, lambda: self._set_ui_state(False))
                return
            self._log(f"Extraction complete. {len(frames)} frames found.")
            
            # 2. Analyze Stream with smart buffering
            if not self.analyzer: raise RuntimeError("Model not ready.")
            
            self.valid_frames = []
            self.after(0, self._clear_gallery)
            
            count = 0
            total_frames = len(frames)
            
            # Smart Buffer Logic
            candidate_buffer = []
            REQUIRED_SEQ = 2 # Minimum sequence length for a valid event
            
            self._update_progress(0.0, f"Analyzing {total_frames} frames...")
            
            for i, frame_path in enumerate(frames):
                if not self.is_processing: 
                    self._log("Processing cancelled during analysis.")
                    break # Exit loop if processing is cancelled
                
                # Predict
                res = self.analyzer.model.predict(frame_path, conf=0.4, verbose=False)[0]
                cls_names = [self.analyzer.model.names[int(b.cls[0])] for b in res.boxes]
                
                # Check Conditions for a "valid" frame
                has_full = "fullscore" in cls_names
                has_score = "score" in cls_names
                has_id = any(c in cls_names for c in ["rank", "song_name", "song_title"])
                
                is_valid_frame = has_full and has_score and has_id
                
                if is_valid_frame:
                    # Score calculation for this frame
                    unique_items = len(set(cls_names))
                    conf_sum = sum([float(b.conf[0]) for b in res.boxes])
                    score_val = (unique_items * 10) + conf_sum # Example scoring
                    
                    candidate_buffer.append({
                        "path": frame_path,
                        "idx": i,
                        "score": score_val,
                        "cls": cls_names
                    })
                else:
                    # Sequence broken? Check buffer for potential event
                    if len(candidate_buffer) >= REQUIRED_SEQ:
                        # Select the best frame from the buffer
                        best = max(candidate_buffer, key=lambda x: x["score"])
                        
                        # Add Event
                        self.valid_frames.append(best["path"])
                        self.after(0, lambda p=best["path"], s=best["score"], idx=best["idx"]: 
                                   self._add_gallery_item(p, s, idx))
                        count += 1
                        
                    candidate_buffer = [] # Reset buffer
                
                # Progress update (less frequent to avoid UI lag)
                if i % 5 == 0: # Update every 5 frames
                    prog = (i + 1) / total_frames
                    self._update_progress(prog, f"Analyzing: {i}/{total_frames} | Found: {count}")

            # After loop, check if there's any remaining sequence in buffer
            if self.is_processing and len(candidate_buffer) >= REQUIRED_SEQ:
                best = max(candidate_buffer, key=lambda x: x["score"])
                self.valid_frames.append(best["path"])
                self.after(0, lambda p=best["path"], s=best["score"], idx=best["idx"]: 
                           self._add_gallery_item(p, s, idx))
                count += 1
            
            self.after(0, self._on_finish)
            
        except Exception as e:
            self._handle_error(e)

    # --- UI Updates ---
    def _update_progress(self, val, text):
        self.after(0, lambda: self.__ui_progress(val, text))
        
    def __ui_progress(self, val, text):
        self.progress_bar.set(val)
        self.lbl_progress_act.configure(text=text)
        # Maybe log verbose updates less frequently to avoid spam?
        # self._log(text) 

    def _show_meta(self, data):
        self.meta_frame.pack(anchor="w", pady=5)
        self.lbl_meta_title.configure(text=data.get('title', 'Unknown'))
        dur = str(datetime.timedelta(seconds=data.get('duration', 0)))
        self.lbl_meta_info.configure(text=f"Uploaded by: {data.get('uploader')} | Duration: {dur}")

    def _clear_gallery(self):
        for w in self.scroll_results.winfo_children(): w.destroy()
        
    def _add_gallery_item(self, path, score, index):
        try:
            # We must load image in main thread or carefully. 
            # This is called via after(), so it's main thread.
            
            # Thumbnail
            img = Image.open(path)
            img.thumbnail((200, 200))
            ctk_img = ctk.CTkImage(img, size=(180, 135))
            
            # Frame Logic
            idx_in_gallery = len(self.scroll_results.winfo_children()) 
            row = idx_in_gallery // 4
            col = idx_in_gallery % 4
            
            card = ctk.CTkFrame(self.scroll_results, fg_color="#222")
            card.grid(row=row, column=col, padx=5, pady=5)
            
            # Button with preview command
            btn = ctk.CTkButton(card, image=ctk_img, text="", width=180, height=135, 
                              fg_color="transparent", hover_color="#333",
                              command=lambda p=path: self.open_preview(p))
            btn.pack(padx=5, pady=5)
            
            info = f"Score: {score:.2f} | Frame: {index}"
            ctk.CTkLabel(card, text=info, font=("Roboto Mono", 10), text_color=COLOR_ACCENT).pack(pady=(0,5))
            
            self._log(f"Event found: Frame {index} (Score {score:.2f})")
            
        except Exception as e:
            self._log(f"Error adding image {path}: {e}")

    def open_preview(self, path):
        # Create Toplevel
        top = ctk.CTkToplevel(self)
        top.title(f"Preview: {os.path.basename(path)}")
        top.geometry("1100x800")
        top.configure(fg_color=COLOR_BG_DARK)
        top.after(100, lambda: top.focus_set()) # Ensure it comes to front

        # Layout
        left_panel = ctk.CTkFrame(top, fg_color="transparent")
        left_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        right_panel = ctk.CTkFrame(top, width=350, fg_color=COLOR_BG_CARD)
        right_panel.pack(side="right", fill="y", padx=10, pady=10)
        
        # Image Display
        try:
            img = Image.open(path)
            # Resize logic for display
            display_w, display_h = 700, 750
            ratio = min(display_w/img.width, display_h/img.height)
            new_size = (int(img.width*ratio), int(img.height*ratio))
            
            tk_img = ctk.CTkImage(img, size=new_size)
            lbl = ctk.CTkLabel(left_panel, image=tk_img, text="")
            lbl.pack(fill="both", expand=True)
        except Exception as e:
            ctk.CTkLabel(left_panel, text=f"Error loading image: {e}").pack()

        # OCR Panel Header
        ctk.CTkLabel(right_panel, text="OCR DATA ANALYSIS", font=("Roboto", 18, "bold"), text_color=COLOR_ACCENT).pack(pady=(20, 10))
        
        # AI Button
        self.btn_run_ai = ctk.CTkButton(right_panel, text="Analyze with AI ✨", command=lambda: run_ai_logic(),
                                      fg_color="#5e35b1", hover_color="#4527a0", state="disabled")
        self.btn_run_ai.pack(fill="x", padx=20, pady=(0, 10))

        # Status Label (Persistent in Right Panel)
        lbl_ocr_status = ctk.CTkLabel(right_panel, text="Running OCR...", font=("Roboto Mono", 12), text_color="gray")
        lbl_ocr_status.pack(pady=(0,10))

        ocr_container = ctk.CTkScrollableFrame(right_panel, fg_color="transparent")
        ocr_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Shared state for this preview
        self.current_preview_results = {}

        def render_results(local_data, ai_data=None):
            # Clear container
            for w in ocr_container.winfo_children(): w.destroy()
            
            # Helper to display a single row (Local vs AI)
            def add_comparison_row(label, key, color="white"):
                local_val = local_data.get(key, "Not detected")
                ai_val = ai_data.get(key, "N/A") if ai_data else None
                crop_path = local_data.get(f"{key}_crop", None)
                
                # Main Card
                card = ctk.CTkFrame(ocr_container, fg_color="#2b2b2b", corner_radius=6)
                card.pack(fill="x", pady=8)
                
                # --- Header ---
                ctk.CTkLabel(card, text=label, font=("Roboto", 11, "bold"), text_color="#aaa").pack(anchor="w", padx=12, pady=(8,2))

                # --- Body Layout ---
                # We use a vertical stack: Top = Local (Img+Txt), Bottom = AI
                
                # 1. LOCAL SECTION
                f_local = ctk.CTkFrame(card, fg_color="transparent")
                f_local.pack(fill="x", padx=10, pady=5)
                
                # Image (Left)
                if crop_path and os.path.exists(crop_path):
                    try:
                        pil_crop = Image.open(crop_path)
                        cw, ch = pil_crop.size
                        ratio = 90 / ch 
                        w = int(cw * ratio)
                        ctk_crop = ctk.CTkImage(pil_crop, size=(w, 90))
                        
                        img_box = ctk.CTkFrame(f_local, fg_color="transparent")
                        img_box.pack(side="left", padx=(0,15))
                        
                        ctk.CTkLabel(img_box, image=ctk_crop, text="").pack()
                        ctk.CTkLabel(img_box, text=f"{cw}x{ch}", font=("Roboto Mono", 8), text_color="gray").pack()
                    except: pass

                # Text & Controls (Right of Image)
                txt_box = ctk.CTkFrame(f_local, fg_color="transparent")
                txt_box.pack(side="left", fill="both", expand=True)
                
                ctk.CTkLabel(txt_box, text="LOCAL OCR", font=("Roboto", 9, "bold"), text_color="gray").pack(anchor="w")
                
                font_size = 24 if key == "score" else 16
                ctk.CTkLabel(txt_box, text=local_val, font=("Roboto Mono", font_size, "bold"), 
                           text_color=color, wraplength=350, justify="left").pack(anchor="w", pady=(0,5))
                
                # Buttons
                btn_row = ctk.CTkFrame(txt_box, fg_color="transparent")
                btn_row.pack(anchor="w")
                ctk.CTkButton(btn_row, text="Copy", width=50, height=22, font=("Roboto", 10), fg_color="#444",
                            command=lambda t=local_val: self.clipboard_clear() or self.clipboard_append(t)).pack(side="left", padx=(0,5))
                if crop_path:
                    ctk.CTkButton(btn_row, text="Img", width=50, height=22, font=("Roboto", 10), fg_color="#27ae60",
                                command=lambda p=crop_path: self._copy_image_to_clipboard(p)).pack(side="left")

                # 2. AI SECTION (Full Width Separator)
                ctk.CTkFrame(card, height=1, fg_color="#444").pack(fill="x", padx=5, pady=5)
                
                f_ai = ctk.CTkFrame(card, fg_color="#222", corner_radius=0)
                f_ai.pack(fill="x", padx=5, pady=(0,5))
                
                ai_header = ctk.CTkFrame(f_ai, fg_color="transparent")
                ai_header.pack(fill="x", padx=8, pady=5)
                ctk.CTkLabel(ai_header, text="AI ANALYSIS ✨", font=("Roboto", 10, "bold"), text_color="#5e35b1").pack(side="left")
                
                if ai_data:
                    # Comparison Logic
                    is_match = str(local_val) == str(ai_val)
                    ai_color = COLOR_SUCCESS if is_match else COLOR_WARNING
                    if str(ai_val).lower() == "null" or ai_val is None: 
                        ai_color = "gray"
                        ai_val = "Not detected / Null"

                    # AI Text Value (Full Width)
                    ctk.CTkLabel(f_ai, text=str(ai_val), font=("Roboto Mono", font_size, "bold"), 
                               text_color=ai_color, wraplength=480, justify="left").pack(padx=10, pady=(0,10), anchor="w")
                    
                    # Optional copy for AI?
                    # ctk.CTkButton(ai_header, text="Copy AI", width=50, ...).pack(side="right") 
                else:
                    ctk.CTkLabel(f_ai, text="Waiting...", font=("Roboto Mono", 11), text_color="#555").pack(padx=10, pady=10, anchor="w")

            add_comparison_row("FULL SCREEN AREA", "fullscore")
            add_comparison_row("SONG TITLE", "song_name")
            add_comparison_row("SCORE", "score", COLOR_SUCCESS)
            add_comparison_row("RANK", "rank", COLOR_WARNING)
            
            # Special Row for AI Difficulty
            if ai_data and ai_data.get("difficulty"):
                 card = ctk.CTkFrame(ocr_container, fg_color="#2b2b2b", corner_radius=6)
                 card.pack(fill="x", pady=8)
                 ctk.CTkLabel(card, text="AI DETECTED DIFFICULTY", font=("Roboto", 10, "bold"), text_color="#888").pack(anchor="w", padx=12, pady=5)
                 ctk.CTkLabel(card, text=ai_data["difficulty"], font=("Roboto Mono", 20, "bold"), text_color="#f39c12").pack(pady=10)

        def run_ocr_logic():
            if not self.analyzer:
                self.after(0, lambda: lbl_ocr_status.configure(text="Analyzer not ready", text_color=COLOR_ERROR))
                return

            try:
                self.current_preview_results = self.analyzer.perform_ocr(path)
                
                def update_ui():
                    lbl_ocr_status.pack_forget()
                    render_results(self.current_preview_results)
                    # Enable AI button if service is ready
                    if self.llm_service:
                        self.btn_run_ai.configure(state="normal")
                    else:
                        self.btn_run_ai.configure(text="AI Not Configured")

                self.after(0, update_ui)
            except Exception as e:
                self.after(0, lambda: lbl_ocr_status.configure(text=f"OCR Error: {e}", text_color=COLOR_ERROR))

        def run_ai_logic():
            if not self.llm_service: return
            
            # Start UI State (Main Thread)
            lbl_ocr_status.configure(text="Contacting AI Service...", text_color=COLOR_ACCENT)
            lbl_ocr_status.pack(pady=10)
            self.btn_run_ai.configure(state="disabled")
            
            def on_ai_complete(results, elapsed):
                lbl_ocr_status.pack_forget()
                self._log(f"AI Analysis finished in {elapsed:.2f}s")
                render_results(self.current_preview_results, results)
                self.btn_run_ai.configure(state="normal")
                self.btn_run_ai.configure(text=f"Analyze with AI ✨ ({elapsed:.1f}s)")
            
            def thread_ai():
                import time
                start_t = time.time()
                try:
                    # Prepare crops dict
                    crops = {}
                    for k in ["song_name", "score", "rank", "fullscore"]:
                        cp = self.current_preview_results.get(f"{k}_crop")
                        if cp: crops[k] = cp
                    
                    ai_results = self.llm_service.analyze_crops(crops)
                    elapsed = time.time() - start_t
                    
                    # Schedule UI Update
                    self.after(0, lambda: on_ai_complete(ai_results, elapsed))
                    
                except Exception as e:
                     self.after(0, lambda: lbl_ocr_status.configure(text=f"AI Error: {e}", text_color=COLOR_ERROR))
                     self.after(0, lambda: self.btn_run_ai.configure(state="normal"))
                     
            threading.Thread(target=thread_ai, daemon=True).start()

        threading.Thread(target=run_ocr_logic, daemon=True).start()

    def _copy_image_to_clipboard(self, path):
        try:
            abs_path = os.path.abspath(path)
            if sys.platform == "darwin":
                # AppleScript to copy image to clipboard
                # 'POSIX file' handles path conversion
                cmd = f'set the clipboard to (read (POSIX file "{abs_path}") as JPEG picture)'
                subprocess.run(["osascript", "-e", cmd])
                self._set_status("Image copied to clipboard!", COLOR_SUCCESS)
            else:
                self._set_status("Clipboard image not supported on this OS", COLOR_WARNING)
        except Exception as e:
            print(f"Clipboard Error: {e}")
            self._set_status("Failed to copy image", COLOR_ERROR)

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{ts}] {msg}\n"
        self.after(0, lambda: self.__ui_log(full_msg))
        
    def __ui_log(self, msg):
        self.log_text.insert("end", msg)
        self.log_text.see("end")

    def _on_finish(self):
        self._set_ui_state(False)
        self._update_progress(1.0, f"Done! Found {len(self.valid_frames)} events.")
        if self.valid_frames:
            self.btn_export.configure(state="normal")
            
    def _set_ui_state(self, processing):
        self.is_processing = processing
        state = "disabled" if processing else "normal"
        self.btn_download.configure(state=state)
        self.btn_file.configure(state=state)
        
        if processing:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()
        else:
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.stop()

    def _handle_error(self, e):
        self._log(f"CRITICAL ERROR: {e}")
        print(f"Error: {e}")
        self.after(0, lambda: messagebox.showerror("Error", str(e)))
        self.after(0, lambda: self._set_ui_state(False))
        
    def _set_status(self, text, color):
        self.after(0, lambda: self.lbl_status.configure(text=text, text_color=color))

    def clean_project_cache(self):
         if messagebox.askyesno("Confirm", "Delete 'output_piu' folder?"):
            try:
                if os.path.exists("output_piu"): shutil.rmtree("output_piu")
                self._clear_gallery()
                self._log("Cache cleared.")
                messagebox.showinfo("Done", "Cache cleared.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def export_frames(self):
        dest = filedialog.askdirectory()
        if dest:
            count = 0
            for p in self.valid_frames:
                try:
                    shutil.copy(p, dest)
                    count += 1
                except: pass
            messagebox.showinfo("Export", f"Exported {count} images.")

if __name__ == "__main__":
    app = PIUApp()
    app.mainloop()
