import torch
import ultralytics
import platform

print("="*50)
print("YOLO Training Environment")
print("="*50)
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Python Version: {platform.python_version()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"Ultralytics Version: {ultralytics.__version__}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

# 检查其他关键包
try:
    import cv2
    print(f"OpenCV Version: {cv2.__version__}")
except ImportError:
    print("OpenCV not installed")