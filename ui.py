import os
import sys
import sqlite3
import cv2
import time
import pytesseract
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QLabel,
    QVBoxLayout, QWidget, QHBoxLayout, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QDateEdit, QMessageBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, QDate, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from ultralytics import YOLO

# Initialize pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class VehicleTypeDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vehicle model not found at {model_path}")
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


class PlateCharacterDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Character model not found at {model_path}")
        self.model = YOLO(model_path)

    def detect_characters(self, plate_img):
        results = self.model(plate_img)
        characters = []

        for result in results:
            for box in result.boxes:
                x1, _, _, _ = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, '?')
                characters.append((x1, label))

        characters.sort(key=lambda x: x[0])
        plate_text = ''.join(label for _, label in characters)
        plate_text = ''.join(c for c in plate_text if c.isalnum())

        if len(plate_text) > 1 and plate_text[0].isalpha() and plate_text[1].isdigit():
            plate_text = plate_text[1:]
        if len(plate_text) > 1 and plate_text[-1].isalpha() and plate_text[-2].isdigit():
            plate_text = plate_text[:-1]

        return plate_text


class DatabaseManager:
    def __init__(self, db_name="LPR.db"):
        self.db_name = db_name
        self._initialize_database()

    def _connect(self):
        return sqlite3.connect(self.db_name)

    def _initialize_database(self):
        conn = self._connect()
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Plates'")
        if not cursor.fetchone():
            self._create_tables()
        else:
            # Check for missing columns
            cursor.execute("PRAGMA table_info(Plates)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'vehicle_type' not in columns:
                cursor.execute("ALTER TABLE Plates ADD COLUMN vehicle_type TEXT DEFAULT 'Unknown'")
            if 'confidence' not in columns:
                cursor.execute("ALTER TABLE Plates ADD COLUMN confidence REAL DEFAULT 0.0")
            if 'image_path' not in columns:
                cursor.execute("ALTER TABLE Plates ADD COLUMN image_path TEXT DEFAULT ''")

        conn.commit()
        conn.close()

    def _create_tables(self):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT UNIQUE,
                owner TEXT,
                vehicle_type TEXT DEFAULT 'Unknown',
                date_time TEXT,
                confidence REAL DEFAULT 0.0,
                image_path TEXT DEFAULT ''
            )
        """)
        conn.commit()
        conn.close()

    def insert_plate(self, plate_text, owner, vehicle_type="Unknown", confidence=0.0, image_path=""):
        conn = self._connect()
        cursor = conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT OR REPLACE INTO Plates (plate, owner, vehicle_type, date_time, confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (plate_text, owner, vehicle_type, date, confidence, image_path))
        conn.commit()
        conn.close()

    def get_owner(self, plate_text):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT owner FROM Plates WHERE plate = ?", (plate_text,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def get_plates_by_date(self, start_date, end_date, plate_filter="", owner_filter="", vehicle_filter="All"):
        conn = self._connect()
        cursor = conn.cursor()

        query = """
            SELECT plate, owner, vehicle_type, date_time, confidence 
            FROM Plates 
            WHERE date(date_time) BETWEEN ? AND ?
        """
        params = [start_date, end_date]

        if plate_filter:
            query += " AND plate LIKE ?"
            params.append(f"%{plate_filter}%")

        if owner_filter:
            query += " AND owner LIKE ?"
            params.append(f"%{owner_filter}%")

        if vehicle_filter != "All":
            query += " AND vehicle_type = ?"
            params.append(vehicle_filter)

        query += " ORDER BY date_time DESC LIMIT 100"

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return results


class CarPlateDetector:
    def __init__(self, plate_model_path, char_model_path=None, vehicle_model_path=None, conf_threshold=0.75,
                 cooldown=10):
        if not os.path.exists(plate_model_path):
            raise FileNotFoundError(f"Plate model not found at {plate_model_path}")

        self.plate_model = YOLO(plate_model_path)
        self.char_detector = PlateCharacterDetector(char_model_path) if char_model_path and os.path.exists(
            char_model_path) else None
        self.vehicle_detector = VehicleTypeDetector(vehicle_model_path) if vehicle_model_path and os.path.exists(
            vehicle_model_path) else None
        self.db = DatabaseManager()
        self.conf_threshold = conf_threshold
        self.cooldown = cooldown
        self.last_detected = {}

    def _should_save_plate(self, plate_text):
        now = time.time()
        if plate_text in self.last_detected and now - self.last_detected[plate_text] < self.cooldown:
            return False
        self.last_detected[plate_text] = now
        return True

    def detect_plate(self, image):
        plate_results = self.plate_model(image)
        vehicle_info = self.vehicle_detector.detect_vehicle(image) if self.vehicle_detector else []

        plates = []

        for result in plate_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf < self.conf_threshold:
                    continue

                roi = image[y1:y2, x1:x2]
                text = self.char_detector.detect_characters(roi) if self.char_detector else ""

                if text and self._should_save_plate(text):
                    owner = self.db.get_owner(text)
                    vehicle_type = self._match_vehicle_type((x1, y1, x2, y2), vehicle_info)

                    if not owner:
                        owner = "Unknown"
                        # In a GUI app, you'd prompt the user here instead

                    self.db.insert_plate(text, owner, vehicle_type, conf)

                    plates.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'text': text,
                        'roi': roi,
                        'owner': owner,
                        'vehicle': vehicle_type
                    })

        return plates

    def _match_vehicle_type(self, plate_bbox, vehicle_info):
        px1, py1, px2, py2 = plate_bbox
        for vehicle in vehicle_info:
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            if vx1 <= px1 <= vx2 and vy1 <= py1 <= vy2:
                return vehicle['label']
        return "Unknown"


class LPRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced LPR System")
        self.setGeometry(100, 100, 1400, 800)

        # Model paths - using relative paths
        self.model_paths = {
            'plate': os.path.normpath("C:/Users/Casper/PycharmProjects/runs/detect/train/weights/best.pt"),
            'char': os.path.normpath("C:/Users/Casper/PycharmProjects/runs/detect/train2/weights/best.pt"),
            'vehicle': os.path.normpath("C:/Users/Casper/runs/detect/train19/weights/best.pt")
        }

        # Create models directory if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")
            QMessageBox.information(self, "Information",
                                    "Created 'models' directory. Please place your model files there.")

        # Initialize detector with error handling
        self.detector = None
        self.initialize_detector()

        self.db = DatabaseManager()
        self.current_image = None
        self.video_capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_frame)

        self.setup_ui()
        self.load_recent_plates()

    def initialize_detector(self):
        try:
            self.detector = CarPlateDetector(
                plate_model_path=self.model_paths['plate'],
                char_model_path=self.model_paths['char'],
                vehicle_model_path=self.model_paths['vehicle'],
                conf_threshold=0.75,
                cooldown=10
            )
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error",
                                 f"Failed to initialize detector: {str(e)}\n\n"
                                 "Application will run with limited functionality.")
            try:
                # Try without vehicle detection
                self.detector = CarPlateDetector(
                    plate_model_path=self.model_paths['plate'],
                    char_model_path=self.model_paths['char'],
                    vehicle_model_path=None,
                    conf_threshold=0.75,
                    cooldown=10
                )
            except Exception as e:
                QMessageBox.critical(self, "Critical Error",
                                     f"Failed to initialize basic detector: {str(e)}")
                self.detector = None

    def setup_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Controls and results
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Model configuration group
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()

        self.plate_model_edit = QLineEdit(self.model_paths['plate'])
        self.char_model_edit = QLineEdit(self.model_paths['char'])
        self.vehicle_model_edit = QLineEdit(self.model_paths['vehicle'])

        model_layout.addWidget(QLabel("Plate Model Path:"))
        model_layout.addWidget(self.plate_model_edit)
        model_layout.addWidget(QLabel("Character Model Path:"))
        model_layout.addWidget(self.char_model_edit)
        model_layout.addWidget(QLabel("Vehicle Model Path:"))
        model_layout.addWidget(self.vehicle_model_edit)

        # Detection settings
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.1, 0.99)
        self.conf_threshold.setValue(0.75)
        self.conf_threshold.setSingleStep(0.05)

        self.cooldown = QSpinBox()
        self.cooldown.setRange(1, 600)
        self.cooldown.setValue(10)

        model_layout.addWidget(QLabel("Confidence Threshold:"))
        model_layout.addWidget(self.conf_threshold)
        model_layout.addWidget(QLabel("Cooldown (seconds):"))
        model_layout.addWidget(self.cooldown)

        update_btn = QPushButton("Update Model Settings")
        update_btn.clicked.connect(self.update_detector_settings)
        model_layout.addWidget(update_btn)

        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # Input selection group
        input_group = QGroupBox("Input Selection")
        input_layout = QVBoxLayout()

        self.input_combobox = QComboBox()
        self.input_combobox.addItems(["Image", "Video", "Camera"])

        self.file_btn = QPushButton("📂 Select File")
        self.file_btn.clicked.connect(self.select_file)

        self.camera_btn = QPushButton("🎥 Start Camera")
        self.camera_btn.clicked.connect(self.start_camera)

        self.process_btn = QPushButton("🔍 Process")
        self.process_btn.clicked.connect(self.process_input)

        input_layout.addWidget(QLabel("Input Type:"))
        input_layout.addWidget(self.input_combobox)
        input_layout.addWidget(self.file_btn)
        input_layout.addWidget(self.camera_btn)
        input_layout.addWidget(self.process_btn)

        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)

        # Filter group
        filter_group = QGroupBox("Filter Options")
        filter_layout = QVBoxLayout()

        # Date filter
        date_layout = QHBoxLayout()
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())

        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("To:"))
        date_layout.addWidget(self.end_date)
        filter_layout.addLayout(date_layout)

        # Plate filter
        self.plate_filter = QLineEdit()
        self.plate_filter.setPlaceholderText("Plate number...")
        filter_layout.addWidget(QLabel("Plate Number:"))
        filter_layout.addWidget(self.plate_filter)

        # Owner filter
        self.owner_filter = QLineEdit()
        self.owner_filter.setPlaceholderText("Owner name...")
        filter_layout.addWidget(QLabel("Owner:"))
        filter_layout.addWidget(self.owner_filter)

        # Vehicle type filter
        self.vehicle_filter = QComboBox()
        self.vehicle_filter.addItems(["All", "Car", "Truck", "Motorcycle", "Bus", "Unknown"])
        filter_layout.addWidget(QLabel("Vehicle Type:"))
        filter_layout.addWidget(self.vehicle_filter)

        # Filter buttons
        filter_btn = QPushButton("🔍 Apply Filters")
        filter_btn.clicked.connect(self.apply_filters)

        reset_btn = QPushButton("🔄 Reset Filters")
        reset_btn.clicked.connect(self.reset_filters)

        filter_layout.addWidget(filter_btn)
        filter_layout.addWidget(reset_btn)

        filter_group.setLayout(filter_layout)
        left_layout.addWidget(filter_group)

        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Plate", "Owner", "Vehicle", "Date", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.doubleClicked.connect(self.show_selected_image)

        left_layout.addWidget(QLabel("Detection Results:"))
        left_layout.addWidget(self.table)

        left_panel.setLayout(left_layout)

        # Right panel - Image display
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")

        # Plate ROI display
        self.plate_label = QLabel()
        self.plate_label.setAlignment(Qt.AlignCenter)
        self.plate_label.setStyleSheet("background-color: black;")

        right_layout.addWidget(QLabel("Input Image:"))
        right_layout.addWidget(self.image_label)
        right_layout.addWidget(QLabel("Detected Plate:"))
        right_layout.addWidget(self.plate_label)

        right_panel.setLayout(right_layout)

        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([500, 900])

        self.setCentralWidget(main_splitter)

        # Update button states
        self.input_combobox.currentTextChanged.connect(self.update_input_buttons)
        self.update_input_buttons(self.input_combobox.currentText())

    def update_input_buttons(self, input_type):
        self.file_btn.setEnabled(input_type in ["Image", "Video"])
        self.camera_btn.setEnabled(input_type == "Camera")
        self.process_btn.setEnabled(input_type in ["Image", "Video"])

    def update_detector_settings(self):
        try:
            self.detector = CarPlateDetector(
                plate_model_path=self.plate_model_edit.text(),
                char_model_path=self.char_model_edit.text(),
                vehicle_model_path=self.vehicle_model_edit.text(),
                conf_threshold=self.conf_threshold.value(),
                cooldown=self.cooldown.value()
            )
            self.show_status_message("Model settings updated successfully")
        except Exception as e:
            self.show_error_message(f"Error updating model: {str(e)}")

    def select_file(self):
        input_type = self.input_combobox.currentText()
        if input_type == "Image":
            path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.png *.bmp)")
        else:  # Video
            path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")

        if path:
            self.current_image = path
            self.display_image(path)
            self.process_btn.setEnabled(True)

    def start_camera(self):
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            self.show_error_message("Failed to open camera")
            return

        self.timer.start(30)  # Update every 30ms (~33fps)
        self.show_status_message("Camera started - press 'Stop Camera' to stop")

    def stop_camera(self):
        if self.video_capture:
            self.timer.stop()
            self.video_capture.release()
            self.video_capture = None
            self.image_label.clear()
            self.show_status_message("Camera stopped")

    def update_video_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.process_frame(frame, live=True)

    def process_input(self):
        input_type = self.input_combobox.currentText()
        if input_type == "Image" and self.current_image:
            image = cv2.imread(self.current_image)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.process_frame(image)
            else:
                self.show_error_message("Failed to load image")
        elif input_type == "Video" and self.current_image:
            self.process_video(self.current_image)

    def process_frame(self, frame, live=False):
        if self.detector is None:
            self.show_error_message("Detector not initialized")
            return

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Perform detection
        plates = self.detector.detect_plate(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Draw detections
        painter = QPainter(pixmap)
        for plate in plates:
            x1, y1, x2, y2 = plate['bbox']
            text = plate['text']
            owner = plate['owner']
            vehicle = plate.get('vehicle', 'Unknown')
            confidence = plate['confidence']

            # If owner is not in database, prompt user
            if owner is None:
                owner, ok = QInputDialog.getText(
                    self, "New Plate Detected",
                    f"Enter owner for plate {text}:",
                    QLineEdit.Normal,
                    ""
                )
                if ok and owner:
                    self.db.insert_plate(text, owner, vehicle, confidence)
                    plate['owner'] = owner  # Update for display
                else:
                    owner = "Unknown"
                    self.db.insert_plate(text, owner, vehicle, confidence)
                    plate['owner'] = owner

            # Draw rectangle
            painter.setPen(QPen(Qt.green, 2))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            # Draw text
            label = f"{text} ({owner}) [{vehicle}] {confidence:.2f}"
            painter.setPen(Qt.green)
            painter.drawText(x1, y1 - 10, label)

            # Display plate ROI
            plate_img = plate['roi']
            if plate_img.size > 0:
                plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                plate_height, plate_width, _ = plate_img.shape
                bytes_per_line = 3 * plate_width
                q_plate_img = QImage(plate_img.data, plate_width, plate_height,
                                     bytes_per_line, QImage.Format_RGB888)
                plate_pixmap = QPixmap.fromImage(q_plate_img)
                self.plate_label.setPixmap(plate_pixmap.scaled(
                    self.plate_label.width(), self.plate_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

        painter.end()
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        if not live:
            self.load_recent_plates()

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.show_error_message("Failed to open video")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.process_frame(frame)

            # Add small delay to allow UI updates
            QApplication.processEvents()

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        self.load_recent_plates()

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.show_error_message("Failed to load image")

    def load_recent_plates(self):
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")

        results = self.db.get_plates_by_date(
            start_date, end_date,
            self.plate_filter.text(),
            self.owner_filter.text(),
            self.vehicle_filter.currentText()
        )

        self.table.setRowCount(len(results))
        for i, (plate, owner, vehicle, date_time, confidence) in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(plate))
            self.table.setItem(i, 1, QTableWidgetItem(owner))
            self.table.setItem(i, 2, QTableWidgetItem(vehicle))
            self.table.setItem(i, 3, QTableWidgetItem(date_time))
            self.table.setItem(i, 4, QTableWidgetItem(f"{confidence:.2f}"))

    def apply_filters(self):
        self.load_recent_plates()

    def reset_filters(self):
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.end_date.setDate(QDate.currentDate())
        self.plate_filter.clear()
        self.owner_filter.clear()
        self.vehicle_filter.setCurrentIndex(0)
        self.load_recent_plates()

    def show_selected_image(self, index):
        plate_text = self.table.item(index.row(), 0).text()
        date_time = self.table.item(index.row(), 3).text()
        self.show_status_message(f"Selected plate {plate_text} from {date_time}")

    def show_status_message(self, message):
        self.statusBar().showMessage(message, 5000)

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
        QMessageBox.information(None, "Information",
                                "Created 'models' directory. Please place your model files there.")

    window = LPRApp()
    window.show()
    sys.exit(app.exec_())