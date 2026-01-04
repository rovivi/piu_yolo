try:
    print("Checking imports...")
    import piu_core
    print("piu_core imported successfully.")
    
    from piu_core import VideoProcessor, FrameAnalyzer
    print("VideoProcessor and FrameAnalyzer imported.")
    
    import piu_app
    print("piu_app imported successfully.")
    
    # Check if classes exist
    vp = VideoProcessor()
    print("VideoProcessor instantiated.")
    
    print("Verification successful!")
except Exception as e:
    print(f"Verification FAILED: {e}")
