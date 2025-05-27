from ultralytics import YOLO
class VehicleTypeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_vehicle(self, image):
        results = self.model(image)
        vehicles = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = result.names[class_id]

                vehicles.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'label': label
                })
        return vehicles

