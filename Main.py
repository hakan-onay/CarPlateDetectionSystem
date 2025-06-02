from CarPlateDetector import CarPlateDetector
from AnalyzeImage import analyze_input
#10.55.48.12_01_20250519124208101_MOTION_DETECTION.jpg
if __name__ == "__main__":
    input_path = "C:/Users/Casper/Desktop/ftp_veri/10.55.48.12_01_20250519102553794_MOTION_DETECTION.jpg"

    plate_model_path = "C:/Users/Casper/PycharmProjects/runs/detect/train/weights/best.pt"
    char_model_path = "C:/Users/Casper/PycharmProjects/runs/detect/train2/weights/best.pt"
    vehicle_model_path = "C:/Users/Casper/runs/detect/train19/weights/best.pt"

    detector = CarPlateDetector(
        plate_model_path=plate_model_path,
        char_model_path=char_model_path,
        vehicle_model_path=vehicle_model_path,
        conf_threshold=0.75,
        cooldown=10
    )

    analyze_input(input_path, detector)
