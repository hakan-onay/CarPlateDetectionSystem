from CarPlateDetector import CarPlateDetector
from AnalyzeInput import analyze_input

if __name__ == "__main__":
    # Path to the input image or video
    input_path = "C:/Users/Hakan/Desktop/car.jpg"

    # Paths to trained YOLO models for plate characters and vehicle type
    plate_model_path = "C:/Users/Hakan/Desktop/PlateModel/weights/best.pt"
    char_model_path = "C:/Users/Hakan/Desktop/CharModel/weights/best.pt"
    vehicle_model_path = "C:/Users/Hakan/Desktop/VehicleModel/weights/best.pt"

    # Initialize the car plate detector
    detector = CarPlateDetector(
        plate_model_path=plate_model_path,
        char_model_path=char_model_path,
        vehicle_model_path=vehicle_model_path,
        conf_threshold=0.75,
        cooldown=10  # seconds to ignore duplicate detections
    )

    # Analyze the given input
    analyze_input(input_path, detector)
