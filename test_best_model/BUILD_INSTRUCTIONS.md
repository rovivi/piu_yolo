# Building PIU Analytics Pro V2

This guide explains how to package the application into a standalone executable for Windows, macOS, and Linux.

## Prerequisites

1.  **Python 3.8+**: Ensure Python is installed.
2.  **Dependencies**: Install the required packages.
    ```bash
    pip install customtkinter ultralytics opencv-python pillow yt-dlp pyinstaller
    ```
    *(Note: On Windows, you might need to install `moviepy` or ensure `ffmpeg` is in your PATH)*
3.  **FFmpeg**: The application requires FFmpeg.
    - **Windows**: Download `ffmpeg.exe` and place it in the same folder as the final executable, or add it to system PATH.
    - **Linux/macOS**: Install via package manager (`apt install ffmpeg` or `brew install ffmpeg`).

## Building the Executable

We use **PyInstaller** to create the standalone application.

### One-Click Build Command

Run this command in your terminal from the project directory:

```bash
pyinstaller --noconfirm --onedir --windowed --name "PIU_Analytics_Pro" \
    --add-data "best.pt:." \
    --hidden-import "PIL._tkinter_finder" \
    piu_app.py
```

### Platform Specifics

#### Windows (.exe)
Run the command above in PowerShell or Command Prompt.
- The output will be in the `dist/PIU_Analytics_Pro` folder.
- You can create an installer using **Inno Setup** (optional) pointing to this folder.

#### macOS (.app)
Run the command in Terminal.
- PyInstaller will generate a `.app` bundle in `dist/`.
- Note: You may need to sign the application if distributing to other users.

#### Linux (Binary)
Run the command in Terminal.
- The output executable will be in `dist/PIU_Analytics_Pro`.

## Troubleshooting

- **Missing `best.pt`**: Ensure the model file `best.pt` is in the same directory as `piu_app.py` before building. The `--add-data` flag attempts to bundle it.
- **CustomTkinter Assets**: Sometimes CustomTkinter json/theme files are not picked up. If the UI looks wrong, use:
    ```bash
    --collect-all customtkinter
    ```
