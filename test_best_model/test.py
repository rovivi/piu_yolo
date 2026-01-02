import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Priority: Local best.pt -> Training folder
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
if not os.path.exists(MODEL_PATH):
    fallback = os.path.join(os.path.dirname(BASE_DIR), 'piu_ia/first_try4/weights/best.pt')
    if os.path.exists(fallback):
        MODEL_PATH = fallback

try:
    model = YOLO(MODEL_PATH)
    class_names = model.names
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- HTML Template (Embedded for simplicity, with premium styling) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PIU Part Detector | AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --glass: rgba(30, 41, 59, 0.7);
            --accent: #8b5cf6;
            --accent-glow: rgba(139, 92, 246, 0.5);
            --text: #f8fafc;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg);
            background-image: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 100%);
            color: var(--text);
            margin: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .header {
            width: 100%;
            padding: 2rem 0;
            text-align: center;
            background: rgba(15, 23, 42, 0.8);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            margin-bottom: 3rem;
        }

        h1 {
            margin: 0;
            font-weight: 700;
            letter-spacing: -1px;
            background: linear-gradient(to right, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
        }

        .main-container {
            width: 90%;
            max-width: 1100px;
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 2rem;
            padding-bottom: 50px;
        }

        @media (max-width: 900px) {
            .main-container { grid-template-columns: 1fr; }
        }

        .drop-zone {
            background: var(--glass);
            border: 2px dashed rgba(139, 92, 246, 0.3);
            border-radius: 24px;
            height: 500px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(12px);
        }

        .drop-zone:hover {
            border-color: var(--accent);
            background: rgba(30, 41, 59, 0.9);
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px var(--accent-glow);
        }

        .drop-zone img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 12px;
        }

        .side-panel {
            background: var(--glass);
            border-radius: 24px;
            padding: 2rem;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.05);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .btn-upload {
            background: linear-gradient(135deg, var(--accent), #d946ef);
            color: white;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            text-align: center;
            cursor: pointer;
            transition: 0.3s;
            border: none;
            width: 100%;
            font-size: 1rem;
        }

        .btn-upload:hover {
            transform: scale(1.02);
            filter: brightness(1.1);
        }

        .detection-item {
            background: rgba(255,255,255,0.03);
            padding: 12px 16px;
            border-radius: 12px;
            border-left: 4px solid var(--accent);
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .loader {
            width: 48px;
            height: 48px;
            border: 5px solid #FFF;
            border-bottom-color: var(--accent);
            border-radius: 50%;
            display: none;
            box-sizing: border-box;
            animation: rotation 1s linear infinite;
        }

        @keyframes rotation {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        #fileInput { display: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>PIU Part Detector</h1>
        <p style="opacity: 0.6">Powered by YOLOv8 Engine</p>
    </div>

    <div class="main-container">
        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <div id="placeholder">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 1rem; opacity: 0.5">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                <p>Arrastra una captura o haz click aquí</p>
            </div>
            <div class="loader" id="loader"></div>
            <img id="resultImage" style="display: none;">
        </div>

        <div class="side-panel">
            <h3 style="margin: 0; font-size: 1.2rem;">Detecciones</h3>
            <div id="resultsList" style="flex-grow: 1;">
                <p style="opacity: 0.4; font-size: 0.9rem;">Esperando datos...</p>
            </div>
            <input type="file" id="fileInput" accept="image/*">
            <button class="btn-upload" onclick="document.getElementById('fileInput').click()">Seleccionar Imagen</button>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const dropZone = document.getElementById('dropZone');
        const loader = document.getElementById('loader');
        const placeholder = document.getElementById('placeholder');
        const resultImage = document.getElementById('resultImage');
        const resultsList = document.getElementById('resultsList');

        fileInput.addEventListener('change', e => {
            if (e.target.files.length) handleFile(e.target.files[0]);
        });

        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.style.borderColor = '#8b5cf6';
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = 'rgba(139, 92, 246, 0.3)';
        });

        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });

        async function handleFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            // UI State
            placeholder.style.display = 'none';
            resultImage.style.display = 'none';
            loader.style.display = 'block';
            resultsList.innerHTML = '';

            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.error) throw new Error(data.error);

                resultImage.src = 'data:image/jpeg;base64,' + data.image;
                resultImage.style.display = 'block';
                
                if (data.detections.length === 0) {
                    resultsList.innerHTML = '<p style="opacity:0.5">Sin detecciones.</p>';
                } else {
                    data.detections.forEach(d => {
                        const div = document.createElement('div');
                        div.className = 'detection-item';
                        div.innerHTML = `<span>${d.name}</span> <span style="font-weight:700; color:#10b981">${Math.round(d.conf * 100)}%</span>`;
                        resultsList.appendChild(div);
                    });
                }
            } catch (err) {
                alert("Error: " + err.message);
                placeholder.style.display = 'block';
            } finally {
                loader.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/detect', methods=['POST'])
def detect():
    if not model:
        return jsonify({'error': 'Modelo no cargado. Verifica best.pt'}), 500
        
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No se recibió imagen'}), 400

    # Read image
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO
    results = model.predict(source=img, conf=0.4, imgsz=1024)
    
    # Plot results
    res_plotted = results[0].plot()
    
    # Extract detection info
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            'name': class_names.get(cls_id, f"ID {cls_id}"),
            'conf': conf
        })

    # Convert to base64 for display
    _, buffer = cv2.imencode('.jpg', res_plotted)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': img_base64,
        'detections': detections
    })

if __name__ == '__main__':
    print("Iniciando Servidor de Detección en: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)