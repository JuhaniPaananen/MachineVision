import cv2
import os
import time
import json

dataset_dir = "dataset" #Tahan tulee dataset kansio, johon se tallentaa kuvat!
labels_dir = "labels" #Tahan tulee labels kansio, johon se tallentaa txt muodossa: <class> <x_center> <y_center> <width> <height>
                                                                                #class on esimerkiksi = 0 eli ensimmäinen class ("Juhani").

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

json_path = "count.json"
if os.path.exists(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        count = data.get("count", 0) 
else:
    count = 0



#Alkuperaisesti count=0, count on jonka se lisaa kuvan ja txt tiedoston nimen peraan tallennus osuudessa.
#count = 23   #Laita tahan mihin kohtaan olet jaanyt eli jos viimeisin kuva ennen lopetusta oli: "image_22.png" sitten pista count=23
            #jolloin se jatkaa eika poista muita kuvia. 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fail")
        break

    cv2.imshow("Ota Kuva Space, lopeta q", frame)

    key = cv2.waitKey(1)
    
    if key % 256 == 32:  # Space key
        timestamp = int(time.time())
        img_name = f"image_{count}.png"
        img_path = os.path.join(dataset_dir, img_name)

        cv2.imwrite(img_path, frame)

        height, width, _ = frame.shape

        #Talla rajaat sen alueen.
        #Rajaat siis alueen, jonka haluat vieda trainerille tunnistettavaksi.
        bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Object") 

        x_min, y_min, w, h = bbox 
        x_max = x_min + w
        y_max = y_min + h

        # Tassa on mita se laskee rajatusta alueesta
        # <x_center> <y_center> <width> <height>
        x_center = (x_min + x_max) / 2 / width
        y_center = (y_min + y_max) / 2 / height
        w_norm = w / width
        h_norm = h / height

        txt_name = f"image_{count}.txt"
        txt_path = os.path.join(labels_dir, txt_name)

        with open(txt_path, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print("Tallennettu!")
        print(f"Image {count} saved at {img_path}")
        print(f"Label saved at {txt_path} -> {x_center:.6f}, {y_center:.6f}, {w_norm:.6f}, {h_norm:.6f}")
        
        count += 1

    elif key & 0xFF == ord('q'):
        json_path = "countLuku.json"
        with open(json_path, "w") as json_file:
            json.dump({"count": count}, json_file)
        break

cap.release()
cv2.destroyAllWindows()