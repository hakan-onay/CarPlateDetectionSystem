from ultralytics import YOLO


class VehicleTypeDetector:
    def __init__(self, model_path):
        # Load Yolo Model
        self.model = YOLO(model_path)

    def detect_vehicle(self, image):

        results = self.model(image)

        # List to store detected vehicles
        vehicles = []

        # Process each detection result
        for result in results:
            for box in result.boxes:
                # Get the bounding box coordinates (top-left and bottom-right)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get the confidence score of the detection
                conf = float(box.conf[0])

                # Get the class ID and corresponding label  car, truck, bus
                class_id = int(box.cls[0])
                label = result.names[class_id]

                # Store the vehicle information in the list
                vehicles.append({
                    'bbox': (x1, y1, x2, y2),  # Bounding box
                    'confidence': conf,  # Confidence
                    'label': label  # Detected class label
                })

        # Return the list of detected vehicles
        return vehicles
