import time
from PyQt5.QtWidgets import QInputDialog, QLineEdit, QMessageBox
from ultralytics import YOLO
from PlateCharacterDetector import PlateCharacterDetector
from VehicleTypeDetector import VehicleTypeDetector
from DatabaseManager import DatabaseManager


class CarPlateDetector:
    def __init__(self, plate_model_path, char_model_path=None, vehicle_model_path=None, conf_threshold=0.75,
                 cooldown=10):
        # Load YOLOv8 plate detection model
        self.plate_model = YOLO(plate_model_path)

        # Load character recognition model (optional)
        self.char_detector = PlateCharacterDetector(char_model_path) if char_model_path else None

        # Load vehicle type classification model (optional)
        self.vehicle_detector = VehicleTypeDetector(vehicle_model_path) if vehicle_model_path else None

        # Initialize local database manager
        self.db = DatabaseManager()

        # Detection parameters
        self.conf_threshold = conf_threshold
        self.cooldown = cooldown

        # Dictionaries for controlling duplicate detections
        self.last_detected = {}      # Keeps track of last detection times
        self.plate_buffer = {}       # Stores recent appearances of plates for confirmation

        # Reference to the main window, needed for user dialogs
        self.parent_window = None

    def set_parent_window(self, window):
        # Assign the main window to show dialogs
        self.parent_window = window

    def _should_save_plate(self, plate_text):
        now = time.time()

        # Filter out short or invalid plate texts
        if not (5 <= len(plate_text) <= 9):
            return False

        # Store current detection time for this plate
        if plate_text not in self.plate_buffer:
            self.plate_buffer[plate_text] = []

        self.plate_buffer[plate_text].append(now)

        # Keep only detections within the last 5 seconds
        recent_times = [t for t in self.plate_buffer[plate_text] if now - t < 5]
        self.plate_buffer[plate_text] = recent_times

        # Save plate only if it appeared at least 3 times in the last 5 seconds
        if len(recent_times) >= 3 and (plate_text not in self.last_detected or now - self.last_detected[plate_text] > self.cooldown):
            self.last_detected[plate_text] = now
            return True

        return False

    def detect_plate(self, image):
        plates = []

        try:
            # Run plate detection
            plate_results = self.plate_model(image, conf=self.conf_threshold)

            # Detect vehicle types if model is available
            vehicle_info = self.vehicle_detector.detect_vehicle(image) if self.vehicle_detector else []

            for result in plate_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # Crop the detected plate region
                    plate_roi = image[y1:y2, x1:x2]

                    # Perform character recognition
                    text = ""
                    if self.char_detector:
                        text = self.char_detector.detect_characters(plate_roi)
                        if not text:
                            continue  # Skip if no characters are detected

                    # Confirm whether to save the detection
                    if text and self._should_save_plate(text):
                        # Try to match with a detected vehicle
                        vehicle_type = self._match_vehicle_type((x1, y1, x2, y2), vehicle_info)

                        # Try to fetch owner info from database
                        owner, vehicle_type_db = self.db.get_owner(text)

                        # If not found, ask the user to input it
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

                        # Append detected plate information
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
        if not vehicle_info:
            return "Unknown"

        px1, py1, px2, py2 = plate_bbox
        plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)

        best_match = None
        best_area = 0

        # Find the vehicle whose bounding box contains the plate center
        for vehicle in vehicle_info:
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            if (vx1 <= plate_center[0] <= vx2 and vy1 <= plate_center[1] <= vy2):
                area = (vx2 - vx1) * (vy2 - vy1)
                if area > best_area:
                    best_area = area
                    best_match = vehicle['label']

        return best_match if best_match else "Unknown"
