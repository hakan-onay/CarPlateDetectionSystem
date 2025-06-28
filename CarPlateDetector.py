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
        self.parent_window = None  # Will be set by MainWindow for GUI dialog use

    def set_parent_window(self, window):
        #Set the parent GUI window for showing input dialogs and warnings.
        self.parent_window = window

    def _should_save_plate(self, plate_text):
        #Prevent saving duplicate plates within a cooldown period.
        now = time.time()
        if plate_text in self.last_detected and now - self.last_detected[plate_text] < self.cooldown:
            return False
        self.last_detected[plate_text] = now
        return True

    def detect_plate(self, image):
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

                        # If vehicle type from DB is available, use it
                        if vehicle_type_db:
                            vehicle_type = vehicle_type_db

                        # Create plate info dictionary
                        plate_info = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'text': text,
                            'roi': plate_roi,
                            'owner': owner or "Not in database",
                            'vehicle': vehicle_type or "Unknown"
                        }

                        plates.append(plate_info)

                        # Show alert if plate is found in database
                        if owner is not None and self.parent_window:
                            alert_msg = f"Vehicle found!\n\nPlate: {text}\nOwner: {owner}\nVehicle Type: {vehicle_type}"
                            QMessageBox.information(self.parent_window, "Vehicle Detected", alert_msg)

        except Exception as e:
            print(f"Detection error: {e}")
            if self.parent_window:
                QMessageBox.warning(self.parent_window, "Detection Error", f"An error occurred: {str(e)}")

        return plates

    def _match_vehicle_type(self, plate_bbox, vehicle_info):

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
