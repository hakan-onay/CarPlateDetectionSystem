import cv2
from ultralytics import YOLO
import torch

def process_video(video_path, model_path="runs/detect/train/weights/best.pt"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Kullanılan cihaz: {device.upper()}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Video açılamadı:", video_path)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = model.predict(source=frame, device=device, imgsz=640, conf=0.4)


        result_frame = results[0].plot()


        cv2.imshow("YOLOv8 - Video (CPU)", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"C:\Users\Hakan\Desktop\TestVideo\TrafficPolice.mp4"
    model_path = r"C:\Users\Hakan\runs\detect\train17\weights\best.pt"
    process_video(video_path, model_path)
