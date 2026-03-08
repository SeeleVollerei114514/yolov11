import torch
from ultralytics import __version__ as ultralytics_version

print("=" * 50)
print("环境验证:")
print("=" * 50)
print(f"PyTorch版本: {torch.__version__}")
print(f"Ultralytics版本: {ultralytics_version}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print("\nGPU信息:")
    print(f"设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
        print(f"  内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("\n警告: 未检测到GPU支持!")
    print("训练和推理将在CPU上进行，速度会非常慢")
    print("建议安装支持CUDA的PyTorch版本")

# 验证YOLO模型加载
try:
    from ultralytics import YOLO

    print("\n尝试加载YOLO模型...")
    model = YOLO('yolov8n.pt')  # 最小的预训练模型
    print("YOLOv8n模型加载成功!")

    # 测试小规模推理
    print("运行测试推理...")
    results = model.predict('ultralytics/assets/bus.jpg', save=True)
    print("测试推理完成! 结果保存在runs/detect/predict目录")

except Exception as e:
    print(f"\n模型加载失败: {str(e)}")
    if "No module named" in str(e):
        print("请确保已安装ultralytics: pip install ultralytics")
    elif "out of memory" in str(e):
        print("GPU内存不足，尝试减小batch大小")