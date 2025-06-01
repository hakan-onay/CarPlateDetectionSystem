import time
from ultralytics import YOLO
from PlateCharacterDetector import PlateCharacterDetector
from VehicleTypeDetector import VehicleTypeDetector
from DatabaseManager import DatabaseManager

class CarPlateDetector:
    def __init__(self, plate_model_path, char_model_path=None, vehicle_model_path=None, conf_threshold=0.75, cooldown=10):
        # Load YOLOv8 models
        self.plate_model = YOLO(plate_model_path)
        self.char_detector = PlateCharacterDetector(char_model_path) if char_model_path else None
        self.vehicle_detector = VehicleTypeDetector(vehicle_model_path) if vehicle_model_path else None
        self.db = DatabaseManager()
        self.conf_threshold = conf_threshold
        self.cooldown = cooldown
        self.last_detected = {}  # To avoid duplicates in short intervals

    def _should_save_plate(self, plate_text):
        now = time.time()
        # Prevent duplicate detections within cooldown period
        if plate_text in self.last_detected and now - self.last_detected[plate_text] < self.cooldown:
            return False
        self.last_detected[plate_text] = now
        return True

    def detect_plate(self, image):
        # Run plate detection model
        plate_results = self.plate_model(image)
        vehicle_info = self.vehicle_detector.detect_vehicle(image) if self.vehicle_detector else []

        plates = []

        for result in plate_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf < self.conf_threshold:
                    continue

                # Crop the detected plate
                roi = image[y1:y2, x1:x2]

                # Run character recognition
                text = self.char_detector.detect_characters(roi) if self.char_detector else ""

                if text and self._should_save_plate(text):
                    print(f"[NEW] Detected plate: {text}")

                    # Query or insert owner info
                    owner = self.db.get_owner(text)
                    if not owner:
                        owner = input(f"Owner for plate '{text}' not found. Enter owner name: ")
                        self.db.insert_plate(text, owner)
                    else:
                        print(f"[INFO] Plate '{text}' already registered to: {owner}")

                    # Match vehicle type (if model available)
                    vehicle_type = self._match_vehicle_type((x1, y1, x2, y2), vehicle_info)

                    plates.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'text': text,
                        'roi': roi,
                        'owner': owner,
                        'vehicle': vehicle_type
                    })

                else:
                    print(f"[SKIPPED] Plate {text or '[empty]'} skipped due to duplication or low confidence")

        return plates

    def _match_vehicle_type(self, plate_bbox, vehicle_info):
        # Check if plate is inside a vehicle bounding box
        px1, py1, px2, py2 = plate_bbox
        for vehicle in vehicle_info:
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            if vx1 <= px1 <= vx2 and vy1 <= py1 <= vy2:
                return vehicle['label']
        return "Unknown"
