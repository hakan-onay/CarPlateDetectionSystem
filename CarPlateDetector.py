import time

from PyQt5.QtWidgets import QInputDialog, QLineEdit, QMessageBox
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
        self.parent_window = None  # Will be set by MainWindow

    def set_parent_window(self, window):
        """Set the parent GUI window for showing dialogs"""
        self.parent_window = window

    def _should_save_plate(self, plate_text):
        now = time.time()
        # Prevent duplicate detections within cooldown period
        if plate_text in self.last_detected and now - self.last_detected[plate_text] < self.cooldown:
            return False
        self.last_detected[plate_text] = now
        return True

    def detect_plate(self, image):
        plates = []
        try:
            # Run plate detection model
            plate_results = self.plate_model(image)
            vehicle_info = self.vehicle_detector.detect_vehicle(image) if self.vehicle_detector else []

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
                        # Get or create owner information
                        owner, vehicle_type_db = self.db.get_owner(text)
                        vehicle_type = self._match_vehicle_type((x1, y1, x2, y2), vehicle_info)

                        # If plate not in database, prompt for owner info
                        if owner is None and self.parent_window:
                            owner, ok = QInputDialog.getText(
                                self.parent_window,
                                "Owner Information",
                                f"Owner for plate '{text}' not found. Enter owner name:",
                                QLineEdit.Normal,
                                ""
                            )
                            if ok and owner:
                                try:
                                    self.db.insert_plate(text, owner, vehicle_type)
                                except Exception as e:
                                    print(f"Database error: {e}")
                                    owner = "Unknown"
                            else:
                                owner = "Unknown"

                        plates.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'text': text,
                            'roi': roi,
                            'owner': owner or "Unknown",
                            'vehicle': vehicle_type or "Unknown"
                        })

        except Exception as e:
            print(f"Detection error: {e}")
            if self.parent_window:
                QMessageBox.warning(self.parent_window, "Detection Error", f"An error occurred: {str(e)}")

        return plates

    def _match_vehicle_type(self, plate_bbox, vehicle_info):
        # Check if plate is inside a vehicle bounding box
        px1, py1, px2, py2 = plate_bbox
        for vehicle in vehicle_info:
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            if vx1 <= px1 <= vx2 and vy1 <= py1 <= vy2:
                return vehicle['label']
        return "Unknown"