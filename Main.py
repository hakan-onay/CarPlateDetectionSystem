from AnalyzeImage import analyze_image
from CarPlateDetector import CarPlateDetector
from VehicleTypeDetector import VehicleTypeDetector

if __name__ == "__main__":

    image_path = "C:/Users/Hakan/Desktop/siyah-plaka.jpg"
    plate_model_path = "C:/Users/Hakan/runs/detect/train17/weights/best.pt"
    vehicle_model_path = "C:/Users/Hakan/runs/detect/train19/weights/best.pt"


    plate_detector = CarPlateDetector(plate_model_path)
    vehicle_detector = VehicleTypeDetector(vehicle_model_path)


    analyze_image(image_path, plate_detector, vehicle_detector)
