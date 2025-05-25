import cv2
import pytesseract

# Tesseract yolu
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class CarPlateDetector:
    def processImage(self, path):
        image = cv2.imread(path)
        if image is None:
            print("Image doesn't read. Check the path:", path)
            return

        # 1. Ön işleme
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 2. Kontur bulma
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        plate_contour = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                plate_contour = approx
                break

        if plate_contour is not None:
            # 3. Dikdörtgeni çiz (doğru evrede burasıdır!)
            cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)

            # 4. Plaka bölgesini kırp (sadece o alanı al OCR için)
            x, y, w, h = cv2.boundingRect(plate_contour)
            plate_roi = gray[y:y+h, x:x+w]

            # 5. OCR
            text = pytesseract.image_to_string(plate_roi, config='--psm 8')
            print("Detected Plate Number:", text)

            # 6. Plaka bölgesini de göster
            cv2.imshow("Detected Plate ROI", plate_roi)
        else:
            print("No plate-like contour found.")

        # 7. Görselleri göster
        cv2.imshow("Original with Rectangle", image)
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
