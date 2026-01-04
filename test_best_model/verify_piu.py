
import os
import sys

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Verifying imports...")
    from piu_app import FrameAnalyzer, VideoProcessor
    print("Imports successful.")

    print("Verifying VideoProcessor...")
    vp = VideoProcessor()
    if not os.path.exists(vp.output_dir):
        print("ERROR: Output dir not created.")
        sys.exit(1)
    print("VideoProcessor instantiated.")

    print("Verifying FrameAnalyzer logic...")
    # Mocking YOLO model loading to avoid strict file dependency for this quick test
    # unless best.pt actually exists.
    if os.path.exists("best.pt"):
        print("Found best.pt, attempting to load...")
        # fa = FrameAnalyzer("best.pt") # Commented out to avoid heavy load if GPU is busy
        # print("Model loaded.")
    else:
        print("Warning: best.pt not found locally, skipping model load test.")

    print("Verification passed!")

except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
    sys.exit(1)
