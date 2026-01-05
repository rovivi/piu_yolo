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

        # Higher resolution for better OCR (Limit to 1080p to keep processing reasonable)
        ydl_opts = {
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
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

        output_pattern = os.path.join(frames_dir, "frame_%04d.png")
        
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

        # Lossless Extraction using PNG
        cmd = [
             "ffmpeg",
            "-threads", "0",
            "-i", video_path,
            "-vf", "fps=2/3",
            "-compression_level", "0", # Faster but still lossless PNG
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
            if f.endswith(".png")
        ])
        return frames

class FrameAnalyzer:
    def __init__(self, model_path: str, output_dir="output_piu"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        try:
            # Initialize OCR Reader
            # Warning suggests "ja" needs "en". detailed_list=["ja", "en"]
            # We try standard mix. If fails, fallback to en.
            print("Attempting to initialize OCR with [en, ko, ja]...")
            self.ocr_reader = easyocr.Reader(['en', 'ko', 'ja'], gpu=torch.cuda.is_available())
            print("OCR Reader initialized with [en, ko, ja] support.")
        except Exception as e_multi:
            print(f"Failed to init multi-language OCR ({e_multi}). Falling back to English only.")
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                print("OCR Reader initialized with [en] support (Fallback).")
            except:
                self.ocr_reader = None
        except Exception as e:
             # Catch-all
            print(f"Warning: OCR Reader failed to initialize completely: {e}")
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
            if cls_name in ["score", "song_name", "song_title", "rank", "fullscore"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Add small padding
                h, w = img.shape[:2]
                pad = 3
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                
                crop = img[y1:y2, x1:x2]
                
                # 1. Try to Save Crop (Independent step)
                crop_path = None
                try:
                    debug_crops_dir = os.path.join(os.path.abspath(self.output_dir), "debug_crops")
                    os.makedirs(debug_crops_dir, exist_ok=True)
                    crop_name = f"{cls_name}_{int(time.time() * 1000)}.png"
                    crop_path = os.path.join(debug_crops_dir, crop_name)
                    # Save as PNG for maximum OCR clarity
                    cv2.imwrite(crop_path, crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    
                    # Store crop path
                    base_key = "song_name" if cls_name in ["song_name", "song_title"] else cls_name
                    ocr_data[f"{base_key}_crop"] = crop_path
                except Exception as e_img:
                    print(f"Warning: Failed to save crop image: {e_img}")

                # 2. Perform OCR
                try:
                    # easyocr.readtext returns list of (bbox, text, confidence)
                    detected_text = self.ocr_reader.readtext(crop, detail=0)
                    if detected_text:
                        combined_text = " ".join(detected_text).strip()
                        
                        # --- Specific Parsing Rules ---
                        if cls_name == "score":
                            # User Requirement: Map 'O'->0, 'I'->1 before filtering numbers
                            # Common OCR misinterpretations for digits
                            txt_upper = combined_text.upper()
                            replacements = {
                                'O': '0', 'D': '0', 'Q': '0',
                                'I': '1', 'L': '1', '|': '1', '!': '1',
                                'S': '5', 'Z': '2', 'B': '8'
                            }
                            for char, digit in replacements.items():
                                txt_upper = txt_upper.replace(char, digit)
                                
                            # Now extract only digits
                            digits = re.findall(r'\d+', txt_upper)
                            combined_text = "".join(digits)
                            
                            # Heuristic: Score <= 1,000,000
                            # If length > 7, it's definitely wrong (e.g. 50974264). 
                            # Usually noise is prefixed (Sequence 1 50974264 -> 150974264)
                            if len(combined_text) > 7:
                                combined_text = combined_text[-7:]
                                
                            # If still > 1,000,000, it might be 890,000 but read as 1890000? 
                            # Or maybe 974264 read as 1974264.
                            # Strict check: Max score 1,000,000
                            try:
                                val = int(combined_text)
                                if val > 1000000:
                                     # If > 1M, try removing leading digits until valid
                                     # but usually just taking last 6 is safe if it was > 1M
                                     if len(combined_text) >= 7:
                                         # valid 1,000,000 is 7 chars. 
                                         # If we have 7 chars and > 1M, implies > 1XXXXXX.
                                         # PIU Scores are often 6 digits unless perfect/high?
                                         # Let's take last 6 digits as a fallback
                                         combined_text = combined_text[-6:]
                            except:
                                pass

                        elif cls_name == "rank":
                            # Validate against possible PIU ranks
                            upper_text = combined_text.upper().replace(" ", "")
                            
                            # Possible Ranks sorted by length to match SSS+ before S
                            possible_ranks = [
                                "SSS+", "SSS", "SS+", "SS", "S+", "S", 
                                "AAA+", "AAA", "AA+", "AA", "A+", "A", 
                                "B+", "B", "C", "D", "F"
                            ]
                            
                            found_rank = None
                            
                            # 1. Exact match attempt
                            if upper_text in possible_ranks:
                                found_rank = upper_text
                            else:
                                # 2. Contains match attempt (greedy)
                                for r in possible_ranks:
                                    if r in upper_text:
                                        found_rank = r
                                        break
                            
                            if found_rank:
                                combined_text = found_rank
                            # If no rank found, we leave combined_text as is (or could set to Unknown)

                        # song_name: Text is kept as is (supports letters, symbols, ko/ja)
                        
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

# --- LLM Integration ---
import requests
import base64
import json

class LLMService:
    def __init__(self, api_url, api_key, model="gpt-4o"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def list_available_models(self) -> List[str]:
        try:
            # Normalize Base URL
            # If user provided full chat path: .../v1/chat/completions -> .../v1/models
            # If user provided base: .../ -> .../v1/models
            
            base = self.api_url.rstrip("/")
            if "/chat/completions" in base:
                base = base.split("/chat/completions")[0]
            
            # Ensure we target /v1/models (common standard)
            if not base.endswith("/v1"):
                # If base is just root (e.g. localhost:1234), assume /v1 needed
                # But some valid paths might be /api/v1... try standard append
                url = f"{base}/v1/models"
            else:
                url = f"{base}/models"
                
            headers = {"Authorization": f"Bearer {self.api_key}"}
            print(f"Fetching models from: {url}")
            
            try:
                res = requests.get(url, headers=headers, timeout=5)
            except:
                # Retry without /v1/models, maybe just /models
                url = f"{base}/models"
                print(f"Retrying fetch models from: {url}")
                res = requests.get(url, headers=headers, timeout=5)

            if res.status_code == 200:
                data = res.json()
                return [m["id"] for m in data.get("data", [])]
            return [f"Error: {res.status_code}"]
        except Exception as e:
            print(f"List Models Error: {e}")
            return []

    def analyze_crops(self, crops: Dict[str, str]) -> Dict:
        """
        Sends provided crops to LLM for analysis.
        Expects crops dict: {'song_name': path, 'score': path, 'rank': path, etc}
        """
        
        # URL Logic: Auto-append correct endpoint if missing
        target_url = self.api_url
        if "/chat/completions" not in target_url:
            # Provide smart defaults for local servers often running on root
            target_url = f"{target_url.rstrip('/')}/v1/chat/completions"
            print(f"Auto-corrected LLM URL to: {target_url}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages_content = [
            {
                "type": "text",
                "text": """You are an expert at analyzing Pump It Up game result screens.
Analyze the provided images and extract the following data in strictly valid JSON format:
{
  "song_name": "The text of the song title (support Korean/Japanese/English)",
  "score": "The numeric score (digits only, 0 to 1000000)",
  "rank": "The rank letter (SSS+, SSS, SS+, SS, S+, S, AAA+, AAA, AA+, AA, A+, A, B+, B, C, D, F)",
  "difficulty": "Extract the difficulty level (e.g. S18, D24, Co-Op). Be flexible and tolerant: if the text is blurry or partial, provide your best guess. Use color hints if possible: Orange/Red=Single(S), Green=Double(D), Yellow=Co-op. Return null ONLY if completely invisible."
}
If an image for a specific field is missing or unclear, set null.
"""
            }
        ]

        # Add available images
        for key, path in crops.items():
            if path and os.path.exists(path):
                try:
                    b64_img = self._encode_image(path)
                    messages_content.append({
                        "type": "text",
                        "text": f"Image for field '{key}':"
                    })
                    messages_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_img}"
                        }
                    })
                except Exception as e:
                    print(f"Error encoding {key}: {e}")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": messages_content
                }
            ],
            "max_tokens": 300
        }
        
        # Only strict JSON mode for official OpenAI models or if user forces it
        # Many local servers (LM Studio, vLLM) fail with 'json_object' on non-supported models
        if "gpt" in self.model.lower():
            payload["response_format"] = { "type": "json_object" }

        try:
            print(f"Sending request to {target_url} with model {self.model}...")
            response = requests.post(target_url, headers=headers, json=payload, timeout=30)
            
            # Retry without response_format if 400 error occurs
            if response.status_code == 400 and "response_format" in payload:
                print("400 Error. Retrying without 'response_format'...")
                del payload["response_format"]
                response = requests.post(target_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                print(f"LLM Error Status: {response.status_code}")
                print(f"Response: {response.text}")
                return {"error": f"HTTP {response.status_code}: {response.text[:100]}"}

            res_json = response.json()
            if 'choices' not in res_json:
                 print(f"Invalid Response Structure: {res_json}")
                 return {"error": "No 'choices' in response"}

            content = res_json['choices'][0]['message']['content']
            
            # Remove ```json formatting if present
            content = content.replace("```json", "").replace("```", "").strip()

            # Ensure we parse JSON
            data = json.loads(content)
            return data
        except Exception as e:
            print(f"LLM Error: {e}")
            return {"error": str(e)}
