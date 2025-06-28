import sys
import cv2
import sqlite3
import qrcode
import pandas as pd
from pyzbar.pyzbar import decode
import webbrowser
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QStackedWidget, QLineEdit, QTextEdit, QFileDialog,
                             QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QInputDialog, QDialog)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor, QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices
from CarPlateDetector import CarPlateDetector
from DatabaseManager import DatabaseManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Plate Recognition System")
        self.setGeometry(100, 100, 1000, 700)

        # Model paths
        self.plate_model_path = "models/PlateModel/weights/best.pt"
        self.char_model_path = "models/CharModel/weights/best.pt"
        self.vehicle_model_path = "models/VehicleModel/weights/best.pt"
        self.db_path = "LPR.db"

        # Detection parameters
        self.conf_threshold = 0.75
        self.cooldown = 10

        # Initialize detector and database
        self.detector = None
        self.db = DatabaseManager(self.db_path)
        self.cap = None
        self.timer = QTimer()

        # Create main widgets
        self.create_navigation()
        self.create_pages()

        # Set up main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.navigation, 1)
        main_layout.addWidget(self.stacked_widget, 4)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect signals
        self.timer.timeout.connect(self.update_frame)

    def create_navigation(self):
        # Create the navigation sidebar
        self.navigation = QWidget()
        self.navigation.setStyleSheet("""
            background-color: #2c3e50;
            color: white;
            font-size: 16px;
        """)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Logo
        logo = QLabel("Plate Recognition")
        logo.setStyleSheet("font-size: 20px; font-weight: bold; padding: 20px;")
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

        # Buttons
        btn_style = """
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                padding: 15px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #3d566e;
            }
        """

        self.btn_detection = QPushButton("Real-time Detection")
        self.btn_detection.setStyleSheet(btn_style)
        self.btn_detection.setIcon(QIcon.fromTheme("camera"))
        self.btn_detection.clicked.connect(lambda: self.switch_page(0))

        self.btn_image = QPushButton("Image Detection")
        self.btn_image.setStyleSheet(btn_style)
        self.btn_image.setIcon(QIcon.fromTheme("image"))
        self.btn_image.clicked.connect(lambda: self.switch_page(1))

        self.btn_video = QPushButton("Video Detection")
        self.btn_video.setStyleSheet(btn_style)
        self.btn_video.setIcon(QIcon.fromTheme("video"))
        self.btn_video.clicked.connect(lambda: self.switch_page(2))

        self.btn_database = QPushButton("Database")
        self.btn_database.setStyleSheet(btn_style)
        self.btn_database.setIcon(QIcon.fromTheme("database"))
        self.btn_database.clicked.connect(lambda: self.switch_page(3))

        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setStyleSheet(btn_style)
        self.btn_settings.setIcon(QIcon.fromTheme("settings"))
        self.btn_settings.clicked.connect(lambda: self.switch_page(4))

        layout.addWidget(self.btn_detection)
        layout.addWidget(self.btn_image)
        layout.addWidget(self.btn_video)
        layout.addWidget(self.btn_database)
        layout.addWidget(self.btn_settings)
        layout.addStretch()

        self.navigation.setLayout(layout)

    def create_pages(self):
        # Create all application pages
        self.stacked_widget = QStackedWidget()

        # Page 1: Real-time Detection
        self.page_detection = self.create_detection_page()
        self.stacked_widget.addWidget(self.page_detection)

        # Page 2: Image Detection
        self.page_image = self.create_image_page()
        self.stacked_widget.addWidget(self.page_image)

        # Page 3: Video Detection
        self.page_video = self.create_video_page()
        self.stacked_widget.addWidget(self.page_video)

        # Page 4: Database
        self.page_database = self.create_database_page()
        self.stacked_widget.addWidget(self.page_database)

        # Page 5: Settings
        self.page_settings = self.create_settings_page()
        self.stacked_widget.addWidget(self.page_settings)

    def create_detection_page(self):
        # Create real-time detection page
        page = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Real-time Plate Detection")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        # Controls
        control_layout = QHBoxLayout()

        self.btn_start = QPushButton("Start Camera")
        self.btn_start.setStyleSheet("padding: 10px; font-size: 16px;")
        self.btn_start.clicked.connect(self.start_camera)

        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.setStyleSheet("padding: 10px; font-size: 16px;")
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_stop.setEnabled(False)

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        layout.addLayout(control_layout)

        # Results
        self.detection_results = QTextEdit()
        self.detection_results.setReadOnly(True)
        self.detection_results.setStyleSheet("font-family: monospace;")
        layout.addWidget(QLabel("Detection Results:"))
        layout.addWidget(self.detection_results)

        page.setLayout(layout)
        return page

    def create_image_page(self):
        # Create image detection page
        page = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Image Plate Detection")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label)

        # Controls
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.setStyleSheet("padding: 10px; font-size: 16px;")
        self.btn_load_image.clicked.connect(self.load_image)

        self.btn_detect_image = QPushButton("Detect Plates")
        self.btn_detect_image.setStyleSheet("padding: 10px; font-size: 16px;")
        self.btn_detect_image.clicked.connect(self.detect_image)
        self.btn_detect_image.setEnabled(False)

        layout.addWidget(self.btn_load_image)
        layout.addWidget(self.btn_detect_image)

        # Results
        self.image_results = QTextEdit()
        self.image_results.setReadOnly(True)
        self.image_results.setStyleSheet("font-family: monospace;")
        layout.addWidget(QLabel("Detection Results:"))
        layout.addWidget(self.image_results)

        page.setLayout(layout)
        return page

    def create_video_page(self):
        # Create video detection page
        page = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Video Plate Detection")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Video display
        self.video_file_label = QLabel()
        self.video_file_label.setAlignment(Qt.AlignCenter)
        self.video_file_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_file_label)

        # Controls
        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_video.setStyleSheet("padding: 10px; font-size: 16px;")
        self.btn_load_video.clicked.connect(self.load_video)

        self.btn_play_video = QPushButton("Play Video")
        self.btn_play_video.setStyleSheet("padding: 10px; font-size: 16px;")
        self.btn_play_video.clicked.connect(self.play_video)
        self.btn_play_video.setEnabled(False)

        self.btn_stop_video = QPushButton("Stop Video")
        self.btn_stop_video.setStyleSheet("padding: 10px; font-size: 16px;")
        self.btn_stop_video.clicked.connect(self.stop_video)
        self.btn_stop_video.setEnabled(False)

        layout.addWidget(self.btn_load_video)
        layout.addWidget(self.btn_play_video)
        layout.addWidget(self.btn_stop_video)

        # Results
        self.video_results = QTextEdit()
        self.video_results.setReadOnly(True)
        self.video_results.setStyleSheet("font-family: monospace;")
        layout.addWidget(QLabel("Detection Results:"))
        layout.addWidget(self.video_results)

        page.setLayout(layout)
        return page

    def create_database_page(self):
        # Create database management page with QR code functionality
        page = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Plate Database")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Search
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by plate, owner or vehicle type...")
        self.search_input.setStyleSheet("padding: 8px; font-size: 14px;")

        self.btn_search = QPushButton("Search")
        self.btn_search.setStyleSheet("padding: 8px; font-size: 14px;")
        self.btn_search.clicked.connect(self.search_database)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.btn_search)
        layout.addLayout(search_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Plate", "Owner", "Vehicle Type", "Date/Time"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setStyleSheet("padding: 8px; font-size: 14px;")
        self.btn_refresh.clicked.connect(self.load_database)

        self.btn_add = QPushButton("Add Plate")
        self.btn_add.setStyleSheet("padding: 8px; font-size: 14px;")
        self.btn_add.clicked.connect(self.add_plate)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setStyleSheet("padding: 8px; font-size: 14px;")
        self.btn_remove.clicked.connect(self.remove_plate)

        self.btn_export_csv = QPushButton("Export to CSV")
        self.btn_export_csv.setStyleSheet("padding: 8px; font-size: 14px;")
        self.btn_export_csv.clicked.connect(self.export_to_csv)


        self.btn_generate_qr = QPushButton("Generate QR")
        self.btn_generate_qr.setStyleSheet("padding: 8px; font-size: 14px;")
        self.btn_generate_qr.clicked.connect(self.generate_qr_code)

        self.btn_scan_qr = QPushButton("Scan QR")
        self.btn_scan_qr.setStyleSheet("padding: 8px; font-size: 14px;")
        self.btn_scan_qr.clicked.connect(self.scan_qr_code)

        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_generate_qr)
        btn_layout.addWidget(self.btn_scan_qr)
        btn_layout.addWidget(self.btn_export_csv)
        layout.addLayout(btn_layout)

        # Load initial data
        self.load_database()

        page.setLayout(layout)
        return page

    def create_settings_page(self):
        # Create settings page
        page = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Model paths
        layout.addWidget(QLabel("Plate Model Path:"))
        self.plate_path_edit = QLineEdit(self.plate_model_path)
        layout.addWidget(self.plate_path_edit)

        layout.addWidget(QLabel("Character Model Path:"))
        self.char_path_edit = QLineEdit(self.char_model_path)
        layout.addWidget(self.char_path_edit)

        layout.addWidget(QLabel("Vehicle Model Path:"))
        self.vehicle_path_edit = QLineEdit(self.vehicle_model_path)
        layout.addWidget(self.vehicle_path_edit)

        layout.addWidget(QLabel("Database Path:"))
        self.db_path_edit = QLineEdit(self.db_path)
        layout.addWidget(self.db_path_edit)

        # Detection parameters
        layout.addWidget(QLabel("Confidence Threshold (0.1-1.0):"))
        self.threshold_edit = QLineEdit(str(self.conf_threshold))
        self.threshold_edit.setValidator(QDoubleValidator(0.1, 1.0, 2))
        layout.addWidget(self.threshold_edit)

        layout.addWidget(QLabel("Cooldown Time (seconds):"))
        self.cooldown_edit = QLineEdit(str(self.cooldown))
        self.cooldown_edit.setValidator(QIntValidator(1, 60))
        layout.addWidget(self.cooldown_edit)

        # Save button
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.setStyleSheet("padding: 10px; font-size: 16px;")
        self.btn_save.clicked.connect(self.save_settings)
        layout.addWidget(self.btn_save)

        # Status
        self.settings_status = QLabel()
        self.settings_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.settings_status)

        page.setLayout(layout)
        return page

    def switch_page(self, index):
        # Switch between pages
        self.stacked_widget.setCurrentIndex(index)

        # Reset navigation buttons
        for btn in [self.btn_detection, self.btn_image, self.btn_video, self.btn_database, self.btn_settings]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #34495e;
                    color: white;
                    border: none;
                    padding: 15px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #3d566e;
                }
            """)

        # Highlight current button
        current_btn = [self.btn_detection, self.btn_image, self.btn_video, self.btn_database, self.btn_settings][index]
        current_btn.setStyleSheet(
            "background-color: #2980b9; color: white; border: none; padding: 15px; text-align: left;")

    # Camera functions
    def start_camera(self):
        # Start camera for real-time detection
        if not self.detector:
            self.initialize_detector()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera!")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start(30)  # Update every 30ms

    def stop_camera(self):
        # Stop camera
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.clear()

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                if hasattr(self, 'current_video_path'):
                    self.stop_video()
                return

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect plates
            plates = self.detector.detect_plate(frame)

            for plate in plates:
                # Draw bounding boxes and text
                x1, y1, x2, y2 = plate['bbox']
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                confidence_percent = plate['confidence'] * 100
                text = f"{plate['text']} ({plate['vehicle']}) - {confidence_percent:.1f}%"
                cv2.putText(rgb_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Update results display
                result_text = (f"Plate: {plate['text']}\nOwner: {plate['owner']}\nVehicle Type: {plate['vehicle']}\nConfidence: {confidence_percent:.1f}%\n\n")

                if hasattr(self, 'current_video_path'):  # Video mode
                    self.video_results.append(result_text)
                else:  # Real-time mode
                    self.detection_results.append(result_text)

            # Display the frame
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            if hasattr(self, 'current_video_path'):  # Video mode
                self.video_file_label.setPixmap(
                    pixmap.scaled(self.video_file_label.width(), self.video_file_label.height(), Qt.KeepAspectRatio))
            else:  # Real-time mode
                self.video_label.setPixmap(
                    pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

        except Exception as e:
            print(f"Frame update error: {e}")
            if hasattr(self, 'current_video_path'):
                self.stop_video()
            else:
                self.stop_camera()
            QMessageBox.critical(self, "Error", f"Error: {str(e)}")

    # Image functions
    def load_image(self):
        # Load an image for plate detection
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
            self.btn_detect_image.setEnabled(True)

    def detect_image(self):
        # Detect plates in loaded image
        if not self.detector:
            self.initialize_detector()

        if hasattr(self, 'current_image_path'):
            image = cv2.imread(self.current_image_path)
            if image is not None:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect plates
                plates = self.detector.detect_plate(image)
                self.image_results.clear()

                for plate in plates:
                    x1, y1, x2, y2 = plate['bbox']
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    confidence_percent = plate['confidence'] * 100
                    text = f"{plate['text']} - {confidence_percent:.1f}%"
                    cv2.putText(rgb_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Update results
                    result_text = f"Plate: {plate['text']}\nOwner: {plate['owner']}\nVehicle Type: {plate['vehicle']}\nConfidence: {confidence_percent:.1f}%\n\n"
                    self.image_results.insertPlainText(result_text)

                    # If plate is not in the database, ask for owner information
                    if plate['owner'] == "Not in database":
                        owner, ok = QInputDialog.getText(
                            self,
                            "Owner Information",
                            f"Plate '{plate['text']}' not found in database.\nPlease enter owner name:",
                            QLineEdit.Normal,
                            ""
                        )
                        if ok and owner:
                            try:
                                self.db.insert_plate(plate['text'], owner, plate['vehicle'])
                                # Update information
                                plate['owner'] = owner
                                # Refresh results
                                self.image_results.clear()
                                for p in plates:
                                    self.image_results.insertPlainText(
                                        f"Plate: {p['text']}\nOwner: {p['owner']}\nVehicle Type: {p['vehicle']}\nConfidence: {p['confidence'] * 100:.1f}%\n\n"
                                    )
                            except Exception as e:
                                QMessageBox.warning(self, "Error", f"Failed to save to database: {str(e)}")

                # Update displayed image
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.image_results.ensureCursorVisible()  # Scroll to the bottom
                self.image_label.setPixmap(
                    pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    # Video functions
    def load_video(self):
        # Load a video file for plate detection
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.current_video_path = file_path
            self.btn_play_video.setEnabled(True)
            self.video_file_label.setText(f"Video loaded: {file_path}")

    def play_video(self):
        # Play the loaded video with plate detection
        if not self.detector:
            self.initialize_detector()

        if self.cap and not hasattr(self, 'current_video_path'):
            self.stop_camera()

        if not hasattr(self, 'current_video_path'):
            QMessageBox.warning(self, "Warning", "Please load a video first!")
            return

        self.cap = cv2.VideoCapture(self.current_video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file!")
            return

        self.btn_play_video.setEnabled(False)
        self.btn_stop_video.setEnabled(True)
        self.timer.start(30)  # Same as camera update rate

    def stop_video(self):
        # Stop video playback
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

        if hasattr(self, 'current_video_path'):
            del self.current_video_path

        self.btn_play_video.setEnabled(True)
        self.btn_stop_video.setEnabled(False)
        self.video_file_label.clear()
        self.video_results.clear()

    # Database functions
    def load_database(self):
        # Load all plates from database into the table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Plates ORDER BY id DESC")
        data = cursor.fetchall()
        conn.close()

        self.table.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, col_data in enumerate(row_data):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))

    def search_database(self):
        # Search plates in database
        search_term = self.search_input.text().strip()
        if not search_term:
            self.load_database()
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM Plates 
            WHERE plate LIKE ? OR owner LIKE ? OR vehicle_type LIKE ? OR date_time LIKE ?
            ORDER BY id DESC
        """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"))
        data = cursor.fetchall()
        conn.close()

        self.table.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, col_data in enumerate(row_data):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))

    def add_plate(self):
        # Add new plate to database
        plate, ok1 = QInputDialog.getText(self, "Add Plate", "Enter plate number:")
        if ok1 and plate:
            owner, ok2 = QInputDialog.getText(self, "Add Owner", "Enter owner name:")
            if ok2 and owner:
                vehicle_type, ok3 = QInputDialog.getText(self, "Add Vehicle Type", "Enter vehicle type:")
                if ok3:
                    try:
                        self.db.insert_plate(plate, owner, vehicle_type)
                        self.load_database()
                        QMessageBox.information(self, "Success", "Plate added successfully!")
                    except sqlite3.IntegrityError:
                        QMessageBox.warning(self, "Error", "Plate already exists in database!")

    def remove_plate(self):
        # Remove selected plate from database
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            plate_id = self.table.item(selected_row, 0).text()
            plate = self.table.item(selected_row, 1).text()

            reply = QMessageBox.question(
                self, "Confirm",
                f"Are you sure you want to delete plate {plate}?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM Plates WHERE id = ?", (plate_id,))
                conn.commit()
                conn.close()

                self.load_database()
                QMessageBox.information(self, "Success", "Plate removed successfully!")

    # QR Code functions
    def generate_qr_code(self):
        # Generate QR code for selected plate
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            # Get plate information (excluding date_time)
            plate_number = self.table.item(selected_row, 1).text()
            owner = self.table.item(selected_row, 2).text()
            vehicle_type = self.table.item(selected_row, 3).text()

            # Create data string (without date)
            data = f"Plate: {plate_number}\nOwner: {owner}\nVehicle: {vehicle_type}"

            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(data.encode("utf-8-sig"))
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")

            # Save QR code
            file_path, _ = QFileDialog.getSaveFileName(self, "Save QR Code", f"{plate_number}.png", "PNG Files (*.png)")
            if file_path:
                img.save(file_path)
                QMessageBox.information(self, "Success", f"QR code saved as {file_path}")
        else:
            QMessageBox.warning(self, "Warning", "Please select a plate from the table first!")

    def scan_qr_code(self):
        # Scan QR code from image file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image with QR Code", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            try:
                # Read the image and decode QR code
                image = cv2.imread(file_path)
                decoded_objects = decode(image)

                if decoded_objects:
                    # Get the first QR code data
                    qr_data = decoded_objects[0].data.decode("utf-8-sig")

                    # Create a dialog to display the QR code information
                    dialog = QDialog(self)
                    dialog.setWindowTitle("Vehicle Information")
                    dialog.setMinimumSize(400, 200)  # Smaller size since we have less info

                    layout = QVBoxLayout()

                    # Add title
                    title = QLabel("Vehicle Information from QR Code")
                    title.setStyleSheet("font-size: 16px; font-weight: bold;")
                    layout.addWidget(title)

                    # Add QR code data
                    text_edit = QTextEdit()
                    text_edit.setPlainText(qr_data)
                    text_edit.setReadOnly(True)
                    layout.addWidget(text_edit)

                    # Add close button
                    btn_close = QPushButton("Close")
                    btn_close.clicked.connect(dialog.close)
                    layout.addWidget(btn_close)

                    dialog.setLayout(layout)
                    dialog.exec_()
                else:
                    QMessageBox.warning(self, "Warning", "No QR code found in the selected image!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read QR code: {str(e)}")

    def generate_shareable_link(self):
        # Generate a shareable HTML page with vehicle info
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            plate_number = self.table.item(selected_row, 1).text()

            html_content = f"""
            <html>
            <head>
                <title>Vehicle Info: {plate_number}</title>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body>
                <h1>Vehicle Information</h1>
                <p>Scan the QR code below with the mobile app:</p>
                <p>Plate: {plate_number}</p>
                <p>More info would be displayed here...</p>
            </body>
            </html>
            """

            file_path, _ = QFileDialog.getSaveFileName(self, "Save Shareable Link", f"{plate_number}.html",
                                                       "HTML Files (*.html)")
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(html_content)

                # Open in default browser
                webbrowser.open(f"file://{file_path}")
                QMessageBox.information(self, "Success", f"Shareable link saved as {file_path}")

    def export_to_csv(self):
        try:
            # Connect to the database and read all data
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM Plates", conn)
            conn.close()

            # Ask user where to save the file
            file_path, _ = QFileDialog.getSaveFileName(self, "Save as CSV", "plates.csv", "CSV Files (*.csv)")
            if file_path:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "Success", "Plate data exported to CSV successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")

    # Settings functions
    def save_settings(self):
        # Save application settings
        self.plate_model_path = self.plate_path_edit.text()
        self.char_model_path = self.char_path_edit.text()
        self.vehicle_model_path = self.vehicle_path_edit.text()
        self.db_path = self.db_path_edit.text()

        # Get threshold and cooldown values
        try:
            self.conf_threshold = float(self.threshold_edit.text())
            if self.conf_threshold < 0.1 or self.conf_threshold > 1.0:
                raise ValueError("Threshold must be between 0.1 and 1.0")
        except ValueError as e:
            self.settings_status.setText(f"Invalid threshold: {str(e)}")
            self.settings_status.setStyleSheet("color: red;")
            return

        try:
            self.cooldown = int(self.cooldown_edit.text())
            if self.cooldown < 1 or self.cooldown > 60:
                raise ValueError("Cooldown must be between 1 and 60 seconds")
        except ValueError as e:
            self.settings_status.setText(f"Invalid cooldown: {str(e)}")
            self.settings_status.setStyleSheet("color: red;")
            return

        # Reinitialize detector with new settings
        self.initialize_detector()

        # Update database connection
        self.db = DatabaseManager(self.db_path)

        self.settings_status.setText("Settings saved successfully!")
        self.settings_status.setStyleSheet("color: green;")

    def initialize_detector(self):
        # Initialize the plate detector with current settings
        try:
            self.detector = CarPlateDetector(
                plate_model_path=self.plate_model_path,
                char_model_path=self.char_model_path,
                vehicle_model_path=self.vehicle_model_path,
                conf_threshold=self.conf_threshold,
                cooldown=self.cooldown
            )
            # Set the parent window for showing dialogs
            self.detector.set_parent_window(self)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize detector: {str(e)}")
            self.detector = None

    def closeEvent(self, event):
        # Handle application close event
        self.stop_camera()
        self.stop_video()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern style

    # Set dark theme palette
    palette = app.palette()
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, Qt.white)
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())