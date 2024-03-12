from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import cv2
import json

def get_yolo_dataset_dicts(img_dir, label_dir, class_names):
    """
    Convert YOLO annotations to Detectron2 format.
    
    Args:
    - img_dir (str): Directory where images are stored.
    - label_dir (str): Directory where YOLO annotation files are stored.
    - class_names (list of str): List of class names.
    
    Returns:
    - dataset_dicts (list of dicts): Data in Detectron2 format.
    """
    dataset_dicts = []
    for filename in os.listdir(img_dir):
        if not filename.endswith('.jpg') and not filename.endswith('.png'):
            continue
        image_path = os.path.join(img_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
        
        record = {}
        height, width = cv2.imread(image_path).shape[:2]
        
        record["file_name"] = image_path
        record["height"] = height
        record["width"] = width
        record["image_id"] = filename
        
        objs = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    class_id, x_center, y_center, w, h = map(float, line.split())
                    
                    x_min = (x_center - w / 2) * width
                    y_min = (y_center - h / 2) * height
                    bbox_w = w * width
                    bbox_h = h * height
                    
                    obj = {
                        "bbox": [x_min, y_min, bbox_w, bbox_h],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": int(class_id),
                    }
                    objs.append(obj)
                    
        record["annotations"] = objs
        dataset_dicts.append(record)

    # save the dataset_dicts to a new file
    with open(f'{label_dir}/dataset_dict.json', 'w') as f:
        json.dump(dataset_dicts, f)
        print(f'dataset_dicts saved to {label_dir}/dataset_dict.json')

    return dataset_dicts

if __name__ == "__main__":
    classes = ['drink', 'laptop', 'utensil'] 
    for d in ["train", "val", "test"]:
        dict = get_yolo_dataset_dicts(f"data/yolo-data/images/{d}", f"data/yolo-data/labels/{d}", classes)
        print(f"dataset_dicts for {d} created")
        DatasetCatalog.register("my_dataset_" + d, lambda d=d: dict)
        MetadataCatalog.get("my_dataset_" + d).set(thing_classes=classes)
        print("DatasetCatalog registered")
        print("MetadataCatalog set")


