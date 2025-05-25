from CarPlateDetector import CarPlateDetector


if __name__ == "__main__":
    detector = CarPlateDetector()
    image_path = "C:/Users/Hakan/Desktop/stanadart-tip-800x800.jpg"  # Buraya görüntü dosyanın tam yolunu yaz
    detector.processImage(image_path)
