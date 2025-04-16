import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = r"C:\Users\juhan\runs2\detect\CarDataset\datasetasd\text_image4.jpg"

if not os.path.exists(image_path):
    print(f"Error: File {image_path} not found!")
else:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)
    print("Extracted Text:\n", text)

    cv2.imshow("Image", image)
    cv2.imshow("Image2", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
