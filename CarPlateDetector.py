import time
from PyQt5.QtWidgets import QInputDialog, QLineEdit, QMessageBox
from ultralytics import YOLO
from PlateCharacterDetector import PlateCharacterDetector
from VehicleTypeDetector import VehicleTypeDetector
from DatabaseManager import DatabaseManager


class CarPlateDetector:
    def __init__(self, plate_model_path, char_model_path=None, vehicle_model_path=None, conf_threshold=0.75,
                 cooldown=10):
        # Load YOLOv8 models
        self.plate_model = YOLO(plate_model_path)
        self.char_detector = PlateCharacterDetector(char_model_path) if char_model_path else None
        self.vehicle_detector = VehicleTypeDetector(vehicle_model_path) if vehicle_model_path else None
        self.db = DatabaseManager()
        self.conf_threshold = conf_threshold  # Only for plate model
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
            # STEP 1: Run plate detection model with our confidence threshold
            plate_results = self.plate_model(image, conf=self.conf_threshold)

            # STEP 2: If vehicle detector exists, run it on the whole image
            vehicle_info = []
            if self.vehicle_detector:
                vehicle_info = self.vehicle_detector.detect_vehicle(image)

            for result in plate_results:
                for box in result.boxes:
                    # Get plate bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # Crop the detected plate
                    plate_roi = image[y1:y2, x1:x2]

                    # STEP 3: Run character recognition on the plate ROI (no threshold check)
                    text = ""
                    if self.char_detector:
                        text = self.char_detector.detect_characters(plate_roi)
                        if not text:  # If no text detected, skip this plate
                            continue

                    if text and self._should_save_plate(text):
                        # STEP 4: Find matching vehicle type for this plate
                        vehicle_type = self._match_vehicle_type((x1, y1, x2, y2), vehicle_info)

                        # Get or create owner information
                        owner, vehicle_type_db = self.db.get_owner(text)

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
                            'roi': plate_roi,
                            'owner': owner or "Unknown",
                            'vehicle': vehicle_type or "Unknown"
                        })

        except Exception as e:
            print(f"Detection error: {e}")
            if self.parent_window:
                QMessageBox.warning(self.parent_window, "Detection Error", f"An error occurred: {str(e)}")

        return plates

    def _match_vehicle_type(self, plate_bbox, vehicle_info):
        """Find the vehicle that contains this plate"""
        if not vehicle_info:
            return "Unknown"

        px1, py1, px2, py2 = plate_bbox
        plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)

        best_match = None
        best_area = 0

        for vehicle in vehicle_info:
            vx1, vy1, vx2, vy2 = vehicle['bbox']

            # Check if plate center is inside vehicle bbox
            if (vx1 <= plate_center[0] <= vx2 and
                    vy1 <= plate_center[1] <= vy2):

                # Calculate area of vehicle
                area = (vx2 - vx1) * (vy2 - vy1)

                # Choose the largest containing vehicle
                if area > best_area:
                    best_area = area
                    best_match = vehicle['label']

        return best_match if best_match else "Unknown"