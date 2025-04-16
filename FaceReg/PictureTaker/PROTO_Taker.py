import cv2
import os
import time

dataset_dir = "dataset"
labels_dir = "labels"

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fail")
        break

    cv2.imshow("Taker", frame)

    key = cv2.waitKey(1)
    
    if key % 256 == 32:
        timestamp = int(time.time())
        img_name = f"image_{count}.png"
        img_path = os.path.join(dataset_dir, img_name)
        
        cv2.imwrite(img_path, frame)
        
        height, width, _ = frame.shape
        
        txt_name = f"image_{count}.txt"
        txt_path = os.path.join(labels_dir, txt_name)
        

        #<class> <x_center> <y_center> <width> <height>
        
        with open(txt_path, "w") as f:
            f.write("0 ")
            f.write("1")

        count += 1

    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
