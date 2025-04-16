import cv2
from ultralytics import YOLO
import json
import os

model = YOLO("best22.pt")

model2 = YOLO('yolov8s.pt')

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, stream=True)

    results2 = model2.track(frame, stream=True) 



    for result in results2:
        classes_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.9: # EHKÄ TÄMÄ => and label == "Person": #Lisaa tahan, etta class on "Person"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                #Tunnista ihminen
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 
                #cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                pass


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if(box.conf > 0.4): ##Saada tahan prosentti maara, jonka todennakoisyydella se tunnistaa sinut.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Olet Juhani {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Et Ole Juhani {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
