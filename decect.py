from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(model=r'D:\ultralytics-8.3.33(yolo)\runs\detect\train4\weights\best.pt')
    model.predict(source=r'D:\ultralytics-8.3.33(yolo)\OIP-C2.jpg',
                  save=True,
                  show=True,
                  )
