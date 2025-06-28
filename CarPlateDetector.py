import time
from PyQt5.QtWidgets import QInputDialog, QLineEdit, QMessageBox
from ultralytics import YOLO
from PlateCharacterDetector import PlateCharacterDetector
from VehicleTypeDetector import VehicleTypeDetector
from DatabaseManager import DatabaseManager


class CarPlateDetector:
    def __init__(self, plate_model_path, char_model_path=None, vehicle_model_path=None, conf_threshold=0.75,
                 cooldown=10):
        # Load the YOLOv8
        self.plate_model = YOLO(plate_model_path)

        # load character recognition model
        self.char_detector = PlateCharacterDetector(char_model_path) if char_model_path else None

        # load vehicle type detection model
        self.vehicle_detector = VehicleTypeDetector(vehicle_model_path) if vehicle_model_path else None

        # Initialize database manager for storing-retrieving plate info
        self.db = DatabaseManager()

        self.conf_threshold = conf_threshold  # Minimum confidence
        self.cooldown = cooldown  # Time limit to avoid duplicate entries
        self.last_detected = {}  # Dictionary to track recently detected plates
        self.detection_counts = {}  # Track how many times each plate was detected
        self.parent_window = None  # Will be set by MainWindow for GUI dialog use
        self.is_real_time = False  # Flag to distinguish between real-time and image modes

    def set_parent_window(self, window):
        # Set the parent GUI window for showing input dialogs and warnings.
        self.parent_window = window

    def set_real_time_mode(self, is_real_time):
        # Set whether we're in real-time/video mode or image mode.
        self.is_real_time = is_real_time
        if not is_real_time:
            self.detection_counts.clear()  # Clear counts when switching to image mode

    def _should_save_plate(self, plate_text):
        # Determine whether to save/process a plate based on detection mode and count.
        now = time.time()

        # Image mode (single processing)
        if not self.is_real_time:
            # If we've already seen this plate in current session, don't process again
            if plate_text in self.detection_counts:
                return False
            self.detection_counts[plate_text] = 1
            return True

        # Real-time/video mode
        if plate_text in self.last_detected:
            # Check if within cooldown period
            if now - self.last_detected[plate_text] < self.cooldown:
                return False

            # Increment detection count
            self.detection_counts[plate_text] = self.detection_counts.get(plate_text, 0) + 1

            # Only process after 3 detections
            if self.detection_counts[plate_text] >= 5:
                self.detection_counts[plate_text] = 0  # Reset counter
                self.last_detected[plate_text] = now
                return True
            return False

        # First time seeing this plate
        self.detection_counts[plate_text] = 1
        self.last_detected[plate_text] = now
        return False

    def detect_plate(self, image):
        # Main detection function: detects plates, reads characters, queries DB, returns list of plate info.
        plates = []  # List to store detected plates and their info

        try:
            # 1: Detect license plates using YOLO with the confidence threshold
            plate_results = self.plate_model(image, conf=self.conf_threshold)

            # 2: Detect vehicles in the whole image if vehicle detector is available
            vehicle_info = []
            if self.vehicle_detector:
                vehicle_info = self.vehicle_detector.detect_vehicle(image)

            # 3: Process each detection result
            for result in plate_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box for plate
                    conf = float(box.conf[0])  # Confidence of the detection

                    # Crop the license plate from the image
                    plate_roi = image[y1:y2, x1:x2]

                    # 4: Perform character recognition
                    text = ""
                    if self.char_detector:
                        text = self.char_detector.detect_characters(plate_roi)
                        if not text:
                            continue  # Skip if no characters detected

                    # Check for duplicate detection and decide whether to save
                    if text and self._should_save_plate(text):
                        # 5: Match the detected plate to a vehicle type
                        vehicle_type = self._match_vehicle_type((x1, y1, x2, y2), vehicle_info)

                        # 6: Get owner info from database
                        owner, vehicle_type_db = self.db.get_owner(text)

                        # 7: If not in DB, ask user for owner name
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

                        # Add result to the list
                        plates.append({
                            'bbox': (x1, y1, x2, y2),  # Bounding box
                            'confidence': conf,  # Detection confidence
                            'text': text,  # Detected license plate text
                            'roi': plate_roi,  # Plate image region
                            'owner': owner or "Unknown",  # Owner name
                            'vehicle': vehicle_type or "Unknown"  # Vehicle type
                        })

        except Exception as e:
            print(f"Detection error: {e}")
            if self.parent_window:
                QMessageBox.warning(self.parent_window, "Detection Error", f"An error occurred: {str(e)}")

        return plates  # Return list of plate dictionaries

    def _match_vehicle_type(self, plate_bbox, vehicle_info):
        # Match a detected plate to a vehicle type based on position.
        if not vehicle_info:
            return "Unknown"

        px1, py1, px2, py2 = plate_bbox
        plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)  # Center point of the plate

        best_match = None
        best_area = 0

        for vehicle in vehicle_info:
            vx1, vy1, vx2, vy2 = vehicle['bbox']

            # Check if plate center lies within vehicle bounding box
            if (vx1 <= plate_center[0] <= vx2 and
                    vy1 <= plate_center[1] <= vy2):

                # Calculate area to choose the largest matching vehicle
                area = (vx2 - vx1) * (vy2 - vy1)

                if area > best_area:
                    best_area = area
                    best_match = vehicle['label']

        return best_match if best_match else "Unknown"