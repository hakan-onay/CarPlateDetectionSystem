import cv2
import pytesseract
from ultralytics import YOLO

# Tesseract ayarı
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class CarPlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_plate(self, image):
        results = self.model(image)
        plates = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                plate_roi = image[y1:y2, x1:x2]
                gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config='--psm 8').strip()

                plates.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'text': text,
                    'roi': plate_roi
                })
        return plates



