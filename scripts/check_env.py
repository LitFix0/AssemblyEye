import platform, sys

print("OS:", platform.platform())
print("Python:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU:", p.name, f"{p.total_memory / 1e9:.1f} GB VRAM")
except ImportError:
    print("torch: not installed")

try:
    import cv2
    print("opencv:", cv2.__version__)
except ImportError:
    print("opencv: not installed")