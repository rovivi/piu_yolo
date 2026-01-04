import os
import shutil
import subprocess
import threading
import time
import re
from typing import List, Dict, Optional, Generator, Callable
import cv2
import torch
from ultralytics import YOLO
import easyocr

# Try to import yt_dlp for video downloading
try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False

class VideoProcessor:
    def __init__(self, output_dir="output_piu"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_url(self, url: str, progress_hook: Optional[Callable] = None) -> Dict:
        """
        Downloads video from URL using yt-dlp, limited to 720p.
        """
        if not HAS_YTDLP:
            raise ImportError("yt_dlp library is missing.")

        # Limit height to 720 to optimize speed/size
        ydl_opts = {
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            'outtmpl': os.path.join(self.output_dir, 'downloaded_video.%(ext)s'),
            'progress_hooks': [progress_hook] if progress_hook else [],
            'quiet': True,
            'no_warnings': True,
            'overwrites': True  # Overwrite existing files
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Metadata fetch
            try:
                info_meta = ydl.extract_info(url, download=False)
            except Exception as e:
                print(f"Warning: Could not fetch metadata: {e}")
                info_meta = {}

            # Download
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            return {
                "path": filename,
                "title": info_meta.get('title', 'Unknown Title'),
                "uploader": info_meta.get('uploader', 'Unknown Author'),
                "duration": info_meta.get('duration', 0),
                "thumbnail": info_meta.get('thumbnail', None)
            }

    def extract_frames(self, video_path: str, progress_callback: Optional[Callable] = None) -> List[str]:
        """
        Extracts frames using ffmpeg.
        Optimization: Threads=auto, FPS=2/3 (one frame every 1.5s).
        Reads stderr to provide real-time frame count.
        """
        frames_dir = os.path.join(self.output_dir, "raw_frames")
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in system PATH.")

        output_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
        
        # Calculate approximate total frames to output
        total_input_frames = 0
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    # FPS ratio: output_fps / input_fps
                    # output_fps = 2/3 = 0.6666
                    ratio = (2.0/3.0) / fps
                    estimated_output = int(total_input_frames * ratio)
                else:
                    estimated_output = 100 # Fallback
            else:
                estimated_output = 100
            cap.release()
        except:
            estimated_output = 100

        # fps=2/3 means 2 frames every 3 seconds -> ~0.666 fps -> 1 frame every 1.5s
        cmd = [
            "ffmpeg",
            "-threads", "0",     # Let ffmpeg choose max threads (all cores)
            "-i", video_path,
            "-vf", "fps=2/3",    # One frame every 1.5 seconds
            "-q:v", "2",         # High quality jpeg
            "-progress", "pipe:1", # Output progress to stdout for easier parsing
            output_pattern
        ]
        
        # We invoke ffmpeg without 'pipe:1' for progress usually, but standard stderr parsing is more reliable for 'frame='
        # Let's revert to standard stderr parsing.
        cmd = [
             "ffmpeg",
            "-threads", "0",
            "-i", video_path,
            "-vf", "fps=2/3",
            "-q:v", "2",
            output_pattern
        ]

        if progress_callback:
            progress_callback(0, 100, f"Initializing extraction (Est. {estimated_output} frames)...")

        # Run ffmpeg with stderr pipe
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        
        frames_extracted = 0
        
        while True:
            # Read line by line
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                # Look for "frame=  123"
                match = re.search(r"frame=\s*(\d+)", line)
                if match:
                    current_frame = int(match.group(1))
                    if current_frame > frames_extracted:
                        frames_extracted = current_frame
                        if progress_callback:
                            percent = min(0.99, frames_extracted / estimated_output)
                            progress_callback(percent, 0, f"Extracting: {frames_extracted}/{estimated_output}")
                            
        if process.returncode != 0:
             # Check if it was just a warning or fatal
             pass 

        # Collect paths
        frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.endswith(".jpg")
        ])
        return frames

class FrameAnalyzer:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.gpu_name = f"Mode: {self.device.upper()}"
        
        if self.device == "cuda":
            try:
                name = torch.cuda.get_device_name(0)
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                if is_rocm:
                    self.gpu_name = f"AMD GPU: {name} (ROCm {torch.version.hip})"
                else:
                    self.gpu_name = f"NVIDIA GPU: {name} (CUDA {torch.version.cuda})"
            except:
                pass
                
        print(f"Analyzer initialized on {self.device} | {self.gpu_name}")
        

        self.stop_requested = False
        
        # Initialize OCR Reader (supports English numbers and latin characters)
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("OCR Reader initialized.")
        except Exception as e:
            print(f"Warning: OCR Reader failed to initialize: {e}")
            self.ocr_reader = None

    def perform_ocr(self, image_path: str) -> Dict[str, str]:
        """
        Runs YOLO to find regions, then OCR on those regions.
        """
        if not self.ocr_reader:
            return {}

        img = cv2.imread(image_path)
        if img is None:
            return {}

        results = self.model.predict(img, conf=0.3, verbose=False)[0]
        ocr_data = {}

        for box in results.boxes:
            cls_idx = int(box.cls[0])
            cls_name = self.model.names[cls_idx]
            
            # Target classes for OCR
            if cls_name in ["score", "song_name", "song_title", "rank"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Add small padding
                h, w = img.shape[:2]
                pad = 3
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                
                crop = img[y1:y2, x1:x2]
                
                try:
                    # easyocr.readtext returns list of (bbox, text, confidence)
                    detected_text = self.ocr_reader.readtext(crop, detail=0)
                    if detected_text:
                        combined_text = " ".join(detected_text).strip()
                        # Use a canonical key for the dictionary
                        key = "song_name" if cls_name in ["song_name", "song_title"] else cls_name
                        ocr_data[key] = combined_text
                except Exception as e:
                    print(f"OCR error on {cls_name}: {e}")

        return ocr_data

    def analyze_stream(self, image_paths: List[str], conf=0.4, iou=0.7) -> Generator[Dict, None, None]:
        """
        Analyzes a sequence of images.
        Yields 'events' (best frame) as they are found.
        
        Strict Logic: 
        - Event candidate MUST have 'score' AND 'fullscore'.
        - Best Frame within window MUST also have 'score' AND 'fullscore'.
        """
        
        FRAME_INTERVAL = 1.5 
        EVENT_WINDOW_SEC = 10.0
        SKIP_WINDOW_SEC = 30.0
        
        FRAMES_IN_EVENT = int(EVENT_WINDOW_SEC / FRAME_INTERVAL) # ~6-7 frames
        FRAMES_TO_SKIP = int(SKIP_WINDOW_SEC / FRAME_INTERVAL)   # ~20 frames
        
        i = 0
        total_frames = len(image_paths)
        
        # Batch size could optimize, but generator pattern is sequential
        
        while i < total_frames:
            if self.stop_requested:
                break
                
            path = image_paths[i]
            
            try:
                # Quick check first frame
                results = self.model.predict(path, conf=conf, iou=iou, verbose=False)
                detections = results[0]
                
                class_names = [self.model.names[int(box.cls[0])] for box in detections.boxes]
                
                # STRICT check for Trigger
                if "fullscore" in class_names and "score" in class_names:
                    
                    # START SEARCH WINDOW
                    # We are in an event. We want the best frame in the next FRAMES_IN_EVENT.
                    
                    best_frame_data = None
                    max_conf = -1.0
                    
                    end_scan = min(i + FRAMES_IN_EVENT, total_frames)
                    
                    # Iterate window (including current 'i')
                    for j in range(i, end_scan):
                        p_curr = image_paths[j]
                        
                        # We need to predict again if j != i (or cache 'i' result)
                        if j == i:
                            res = results
                        else:
                            res = self.model.predict(p_curr, conf=conf, iou=iou, verbose=False)
                        
                        det = res[0]
                        c_names = [self.model.names[int(b.cls[0])] for b in det.boxes]
                        
                        # STRICT Check for Best Frame Candidate
                        if "fullscore" in c_names and "score" in c_names:
                            # Calculate score (sum of confidences or mean?)
                            # Sum is good: higher confidence on both boxes = better
                            curr_conf = sum([float(b.conf[0]) for b in det.boxes])
                            
                            if curr_conf > max_conf:
                                max_conf = curr_conf
                                best_frame_data = {
                                    "path": p_curr,
                                    "index": j,
                                    "score": curr_conf
                                }
                    
                    # Did we find a valid best frame?
                    if best_frame_data:
                        yield best_frame_data
                        # Jump ahead from the found best frame + skip
                        i = best_frame_data["index"] + FRAMES_TO_SKIP
                    else:
                        # Found a trigger but no "better" subsequent frame (or all failed strict check?)
                        # If trigger passed strict check, then 'i' itself should have been caught as best_frame_data at least.
                        # So this else block is rare.
                        i += 1
                        
                    continue
            
            except Exception as e:
                print(f"Error analyzing frame {path}: {e}")
            
            i += 1
