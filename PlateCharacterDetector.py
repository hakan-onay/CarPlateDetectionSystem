from ultralytics import YOLO

class PlateCharacterDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_characters(self, plate_img):
        results = self.model(plate_img)
        characters = []

        for result in results:
            for box in result.boxes:
                x1, _, _, _ = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, '?')
                characters.append((x1, label))

        characters.sort(key=lambda x: x[0])
        plate_text = ''.join(label for _, label in characters)
        plate_text = ''.join(c for c in plate_text if c.isalnum())

        if len(plate_text) > 1 and plate_text[0].isalpha() and plate_text[1].isdigit():
            plate_text = plate_text[1:]
        if len(plate_text) > 1 and plate_text[-1].isalpha() and plate_text[-2].isdigit():
            plate_text = plate_text[:-1]

        return plate_text
