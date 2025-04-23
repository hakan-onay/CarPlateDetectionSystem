from CarPlateDetector import CarPlateDetector


if __name__ == "__main__":
    detector = CarPlateDetector()
    image_path = "C:/Users/Hakan/Desktop/car.jpg"  
    detector.processImage(image_path)
