from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'yolov11n.yaml')
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=100,
                single_cls=True,
                batch=8,  # 减小batch size避免内存不足
                workers=4,  # 减少工作线程
                device='cpu',  # 关键修改：使用CPU
                )