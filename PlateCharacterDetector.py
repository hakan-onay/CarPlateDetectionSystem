from ultralytics import YOLO

class PlateCharacterDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_characters(self, plate_img):
        # Detect characters on cropped plate
        results = self.model(plate_img)
        characters = []

        for result in results:
            for box in result.boxes:
                x1 = int(box.xyxy[0][0])
                class_id = int(box.cls[0])
                label = result.names.get(class_id, '?')
                characters.append((x1, label))

        # Sort characters by horizontal position
        characters.sort(key=lambda x: x[0])

        # Combine characters into string
        plate_text = ''.join(label for _, label in characters)
        plate_text = ''.join(c for c in plate_text if c.isalnum())  # Clean non-alphanumerics

        # Optional: remove edge letters if misclassified
        if len(plate_text) > 1 and plate_text[0].isalpha() and plate_text[1].isdigit():
            plate_text = plate_text[1:]
        if len(plate_text) > 1 and plate_text[-1].isalpha() and plate_text[-2].isdigit():
            plate_text = plate_text[:-1]

        return plate_text
