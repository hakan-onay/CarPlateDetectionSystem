import cv2

def analyze_input(input_path, plate_detector):
    # Determine input type
    is_video = isinstance(input_path, int) or input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

    if is_video:
        # Open video or camera stream
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Failed to open video or camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect plates from frame
            plates = plate_detector.detect_plate(frame)

            # Print results
            for plate in plates:
                text = plate['text']
                owner = plate['owner']
                vehicle = plate.get('vehicle', 'Unknown')
                print(f"Detected plate: {text} | Owner: {owner} | Vehicle: {vehicle}")

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    elif is_image:
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            print("Could not read image:", input_path)
            return

        # Detect plates from image
        plates = plate_detector.detect_plate(image)

        # Print results
        for plate in plates:
            text = plate['text']
            owner = plate['owner']
            vehicle = plate.get('vehicle', 'Unknown')
            print(f"Detected plate: {text} | Owner: {owner} | Vehicle: {vehicle}")
    else:
        print("Unsupported file format:", input_path)
