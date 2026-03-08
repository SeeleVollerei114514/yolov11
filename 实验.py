import cv2

def resize_video(input_path, output_path, target_width, target_height):
    cap = cv2.VideoCapture(input_path)
    # 获取原视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 定义编码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 调整帧大小
        resized_frame = cv2.resize(frame, (target_width, target_height))
        out.write(resized_frame)

    cap.release()
    out.release()

# 示例：将视频调整为1080p
resize_video('demo.mp4', 'output_1080p.avi', 1920, 1080)
# 示例：将视频调整为480p
resize_video('demo.mp4', 'output_480p.avi', 854, 480)