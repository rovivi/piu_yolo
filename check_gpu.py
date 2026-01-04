import os
# Set environment variables BEFORE importing torch
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["HSA_ENABLE_SDMA"] = "0"

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA/ROCm available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Device properties: {props}")
    if hasattr(torch.version, 'hip'):
        print(f"HIP (ROCm) version: {torch.version.hip}")
else:
    print("CUDA/ROCm not available to PyTorch.")

# Check environment variables
print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
