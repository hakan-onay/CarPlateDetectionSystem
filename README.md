
# 🚗 Car Plate Recognition System

This is an advanced License Plate Recognition (LPR) system powered by YOLOv8 and PyQt5. It can detect vehicle license plates in real-time from camera feed, static images, and video files. It also supports character recognition, vehicle type classification, and integrates with a local SQLite database. Additionally, the system can generate and scan QR codes for plate records.

---

## 🌟 Features

- **Real-Time Detection**: Detect and recognize license plates using your webcam.
- **Image & Video Detection**: Analyze uploaded images or video files for plate recognition.
- **Plate Character Recognition**: Use a trained YOLO model to segment and recognize plate characters.
- **Vehicle Classification**: Distinguish between different vehicle types (e.g., car, truck, bus).
- **SQLite Database**: Automatically log detected plates, owners, vehicle types, and timestamps.
- **GUI with PyQt5**: Intuitive interface to interact with the system.
- **QR Code Integration**: Generate and scan QR codes for any plate entry.
- **CSV Export**: Export all plate logs as a CSV file.

---

## 🧠 Technologies Used

- **Python 3.8+**
- **YOLOv8 (Ultralytics)** - Object detection
- **OpenCV** - Image processing
- **PyQt5** - GUI framework
- **SQLite3** - Lightweight database
- **pandas** - Data manipulation and CSV export
- **qrcode & pyzbar** - QR code generation and decoding

---

## 📦 Folder Structure

```
project/
├── MainWindow.py
├── CarPlateDetector.py
├── PlateCharacterDetector.py
├── VehicleTypeDetector.py
├── DatabaseManager.py
├── models/
│   ├── PlateModel/weights/best.pt
│   ├── CharModel/weights/best.pt
│   └── VehicleModel/weights/best.pt
└── LPR.db

```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/hakan-onay/CarPlateDetectionSystem.git
cd CarPlateDetectionSystem
```

### 2. Install Requirements

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix or Mac
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python MainWindow.py
```

---

## 🧪 Model Requirements

Ensure you have trained YOLOv8 models for:
- **License Plate Detection**
- **Character Recognition**
- **Vehicle Type Detection**

Place them inside the `models/` folder in the appropriate subdirectories.

---

## 📊 Example Plate Record (Database)

| Plate     | Owner       | Vehicle Type | Date Time           |
|-----------|-------------|--------------|---------------------|
| 34ABC123  | Hakan Onay  |     Car      | 2025-06-28 14:32:45 |

---

## 📸 Sample Use Cases

- Automatic gate entry systems
- Parking management
- Campus or company vehicle tracking
- Traffic law enforcement systems

---

## 🔒 Notes

- Accuracy of recognition depends on model quality and image clarity.
- QR code generation only encodes plate, owner, and vehicle type.

---

## 🧠 Future Enhancements

- Export results as PDF
- Multi-language support
- REST API for cloud sync
- Plate format validation based on country rules

---

## 🙋‍♂️ Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyQt5](https://pypi.org/project/PyQt5/)
- [OpenCV](https://opencv.org/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
