import cv2

def analyze_image(image_path, plate_detector, vehicle_detector):
    image = cv2.imread(image_path)
    if image is None:
        print("Image can't be read:", image_path)
        return

    plates = plate_detector.detect_plate(image)
    vehicles = vehicle_detector.detect_vehicle(image)


    for plate in plates:
        x1, y1, x2, y2 = plate['bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{plate['text']} ({plate['confidence']:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle['bbox']
        label = f"{vehicle['label']} ({vehicle['confidence']:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Detected Vehicles and Plates", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()