
# 🚘 License Plate Recognition System (LPR)

A real-time, multi-functional License Plate Recognition system powered by YOLOv8, OpenCV, and PyQt5. This system can detect, segment, and recognize license plates from images, videos, or live camera feeds with integrated database and user-friendly GUI.

---

## 🌟 Key Features

- 📸 **Real-Time Detection** via camera feed  
- 🖼️ **Offline Recognition** from images and videos  
- 🧠 **Multi-Stage Detection Pipeline**:
  - License plate localization
  - Character segmentation & OCR
  - Vehicle type classification (car, truck, bus)
- 🗃️ **SQLite Integration** for storage and search
- 🖥️ **GUI Interface** built with PyQt5
- ⚙️ **Configurable Settings**: detection thresholds, cooldown time, etc.

---

## 🛠 Tech Stack

### Core Technologies
- **Python 3.8+**
- **YOLOv8** – Custom-trained models for detection and classification
- **OpenCV** – Image and video processing
- **PyQt5** – GUI development
- **SQLite** – Lightweight embedded database

### Models Used
- 🔳 **Plate Detection** - YOLOv8
- 🔡 **Character Recognition** - YOLOv8 OCR
- 🚚 **Vehicle Classification** - YOLOv8

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.8+
- NVIDIA GPU (CUDA supported) recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/license-plate-recognition.git
cd license-plate-recognition

# Create virtual environment
python -m venv venv
# Activate (Windows)
.env\Scriptsctivate
# or Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Model Weights
Place your trained weights in the following structure:

```
models/
├── CharModel/weights/best.pt
├── PlateModel/weights/best.pt
└── VehicleModel/weights/best.pt
```

---

## 💻 Usage

### Launch the Application

```bash
python main.py
```

### Application Modes

#### 🎥 Real-Time Detection
- Use your camera for live recognition
- Automatically identifies and logs license plates
- Optional vehicle classification

#### 🖼 Image Processing
- Load and process images
- Recognize and store detected plates

#### 📼 Video Processing
- Frame-by-frame plate recognition
- Visual output with annotated results

#### 📋 Database Management
- View, add, edit, and delete records
- Full search functionality for detected plates

#### ⚙️ System Settings
- Modify model paths
- Set detection confidence thresholds
- Define cooldown period between detections

---

## 🏗 System Architecture

```
[Camera/Input] --> [YOLOv8 Detection] --> [Character Segmentation & OCR]
                                      ↘︎
                          [Vehicle Type Classifier]
                                      ↘︎
                           [Database & GUI Display]
```

---

## 📂 File Structure

```
LPR-System/
├── main.py                   # Entry point for GUI
├── CarPlateDetector.py       # Plate detection pipeline
├── PlateCharacterDetector.py # OCR for plate text
├── VehicleTypeDetector.py    # Classify vehicle type
├── DatabaseManager.py        # Database handling (CRUD)
├── models/                   # YOLOv8 model weights
├── LPR.db                    # SQLite DB file
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo  
2. Create a new branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

---


## ✨ Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyQt5](https://riverbankcomputing.com/software/pyqt)

---

> Made with ❤️ for learning, research, and practical application in vehicle license plate recognition.
