import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np

import re
from collections import Counter
from rapidfuzz import fuzz
import json
import hashlib
import datetime, json
from datetime import datetime

otos_lista = []
cleaned_results = []

def save_to_json(plate):
    if plate != None:

        data = {"plate": plate, "timeStamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}

        with open("saved_plates.json", "w") as f:
            json.dump(data, f, indent=4)

def most_likely_plate(ocr_results, similarity_threshold=90):
    plate_regex = re.compile(r'^[A-Z]{3}-\d{2,3}$')

    # Step 1: Clean and filter valid-ish plates
    for text in ocr_results:
        cleaned = re.sub(r'[^A-Z0-9\-]', '', text.upper())
        if plate_regex.match(cleaned):
            cleaned_results.append(cleaned)

    if not cleaned_results:
        return None

    # Step 2: Fuzzy match and merge near-duplicates
    grouped = []

    for plate in cleaned_results:
        matched = False
        for group in grouped:
            if fuzz.ratio(plate, group[0]) >= similarity_threshold:
                group.append(plate)
                matched = True
                break
        if not matched:
            grouped.append([plate])

    # Step 3: Count the biggest group
    best_group = max(grouped, key=len)
    most_common = Counter(best_group).most_common(1)[0][0]
    return most_common

grouped = []


def top_n_likely_plates(ocr_results, n=3, similarity_threshold=90):
    plate_regex = re.compile(r'^[A-Z]{3}-\d{2,3}$')

    # Step 1: Clean and filter valid-ish plates
    for text in ocr_results:
        cleaned = re.sub(r'[^A-Z0-9\-]', '', text.upper())
        if plate_regex.match(cleaned):
            cleaned_results.append(cleaned)

    if not cleaned_results:
        return None

    # Step 2: Fuzzy match and merge near-duplicates
    #grouped = []

    for plate in cleaned_results:
        matched = False
        for group in grouped:
            if fuzz.ratio(plate, group[0]) >= similarity_threshold:
                group.append(plate)
                matched = True
                break
        if not matched:
            grouped.append([plate])
            
    # Step 3: Count the occurrences in the groups
    all_valid_plates = []
    for group in grouped:
        all_valid_plates.extend(group)

    plate_counts = Counter(all_valid_plates)
    
    # Get the top n plates based on frequency
    most_common_plates = plate_counts.most_common(n)
    return [plate[0] for plate in most_common_plates]

# Windows Only 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#model = YOLO('yolov8s.pt')
model = YOLO('Kilvet.pt')

cap = cv2.VideoCapture(0)

frame_count = 0

frame_texts = []
frame_stopRead = 0

best_match = ''

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    if len(cleaned_results) > 10:
        grouped.clear()
        cleaned_results.clear()

    frame_count += 1
    if frame_count % 5 == 0:
        frame_count = 0
        frame = cv2.resize(frame, (1920, 1080))

        results = model(frame)

        boxes = results[0].boxes.xywh  # (x_center, y_center, width, height)
        class_ids = results[0].boxes.cls
        confidences = results[0].boxes.conf

        for i, box in enumerate(boxes):
            confidence = confidences[i]
            class_id = int(class_ids[i])

            if confidence > 0.3:
                if class_id == 0: # Kilvet.pt:n ainoa luokka
                    x_center, y_center, width, height = box

                    # Convert from xywh (center, width, height) to xyxy (x1, y1, x2, y2)
                    x1 = int((x_center - width / 2))
                    y1 = int((y_center - height / 2))
                    x2 = int((x_center + width / 2))
                    y2 = int((y_center + height / 2))

                    # This is cropped frame from cellphone.
                    cropped = frame[y1:y2, x1:x2]

                    # Preprocess the cropped image
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    denoised = cv2.GaussianBlur(binary, (5, 5), 0)

                    text = pytesseract.image_to_string(denoised, config=r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')

                    # More can be added to for proper validation.
                    filtered_text = text.replace('I', '1').replace('l', '1').replace('O', '0').replace(' ', '').replace('|', '1')
                    frame_texts.append(filtered_text)

                    teksti = most_likely_plate(frame_texts)

                    if teksti is not '':
                        otos_lista.append(teksti)

                    if teksti is '' :
                        otos_lista.clear()
                        grouped.clear()
                        if len(cleaned_results) > 10:
                            cleaned_results.clear()

                        
                    
                    if len(otos_lista) > 10:
                        save_to_json(teksti)
                        otos_lista.clear()


                    print(f"Top 3: {top_n_likely_plates(frame_texts, n=3)}")
                    print(f"Paras vaihtoehto: {teksti}")
                    print(f"Kuvan teksti otettu: {filtered_text}")

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2
                    color = (0, 255, 0)
                    thickness = 2

                    position = (10, 30)  # (x, y)

                    #teksti += str(len(cleaned_results))

                    # Put the text on the image
                    cv2.putText(frame, teksti, position, font, font_scale, color, thickness)
                    
                    # Optional
                    # cv2.imshow("Cropped", cropped)

        cv2.imshow('Real-time Object Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
