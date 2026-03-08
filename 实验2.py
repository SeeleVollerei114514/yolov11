from ultralytics import YOLO
import cv2

model=YOLO('D:/ultralytics-8.3.33(yolo)/yolo11n.pt')
# video_source=0#默认摄像头，外接摄像头为1
video_source='output_1080p.avi'#本地读取

cap=cv2.VideoCapture(video_source)
if not cap.isOpened():
    print('failed, please restarted.')
    exit()
while True:
    ret,frame=cap.read()
    if not ret:
        print('fail read , please retry')
        break
    frame=cv2.resize(frame,(0,0),fx=1.0,fy=1.0)#fx=0.5,fy=0.5
    result=model(frame,classes=[2],conf=0.5)
    annotated_frame=result[0].plot()
    fps=cap.get(cv2.CAP_PROP_FPS)
    #（字体、大小、颜色、厚度）
    cv2.putText(annotated_frame,f'Fps:{int(fps)}',(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow('实时车辆检测',annotated_frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print('video annonate finished')