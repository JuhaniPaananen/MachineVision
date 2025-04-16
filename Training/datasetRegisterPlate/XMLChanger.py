import os
import xml.etree.ElementTree as ET

# CAUTION ONLY FOR DATASET WITH ONE CLASS

classes = ["licence"] # Class name

xml_dir = "annotations"
txt_dir = "labels"
os.makedirs(txt_dir, exist_ok=True)

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    return x_center * dw, y_center * dh, width * dw, height * dh

for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    txt_filename = os.path.splitext(xml_file)[0] + ".txt"
    with open(os.path.join(txt_dir, txt_filename), "w") as out_file:
        for obj in root.iter("object"):
            cls_name = obj.find("name").text
            if cls_name not in classes:
                continue
            cls_id = classes.index(cls_name)
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymax").text),
            )
            bb = convert_bbox((w, h), b)
            out_file.write(f"{cls_id} {' '.join([str(round(val, 6)) for val in bb])}\n")