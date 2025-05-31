import cv2

def analyze_input(input_path, plate_detector):
    is_video = isinstance(input_path, int) or input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

    if is_video:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Failed to open video or camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            plates = plate_detector.detect_plate(frame)
            for plate in plates:
                x1, y1, x2, y2 = plate['bbox']
                text = plate['text']
                owner = plate['owner']
                vehicle = plate.get('vehicle', 'Unknown')
                label = f"{text} ({owner}) [{vehicle}]"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Plate Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif is_image:
        image = cv2.imread(input_path)
        if image is None:
            print("Could not read image:", input_path)
            return

        plates = plate_detector.detect_plate(image)
        for i, plate in enumerate(plates):
            x1, y1, x2, y2 = plate['bbox']
            text = plate['text']
            owner = plate['owner']
            vehicle = plate.get('vehicle', 'Unknown')
            label = f"{text} ({owner}) [{vehicle}]"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f"Plate {i + 1}", plate['roi'])

        cv2.imshow("Detection Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Unsupported file format:", input_path)
