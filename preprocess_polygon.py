import os
import numpy as np
from PIL import Image
from pycocotools.mask import encode, toBbox
import tqdm
import pickle
from detectron2.structures import BoxMode
import cv2
import pycocotools.mask as mask_util

def polygonFromMask(maskedArr): # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py

    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = mask_util.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = mask_util.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = mask_util.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0] #, [x, y, w, h], area


def preprocess_mask(train_set_path="dataset/train"):
    """
    Compress the mask images into rle format, save the annotations for each image in a pickle file.
    """
    for img_folder in tqdm.tqdm(os.listdir(train_set_path)):
        annotations = []
        mask_folder = os.path.join(train_set_path, img_folder, "masks")
        for mask_file in os.listdir(mask_folder):
            if mask_file.startswith("."):
                continue
            mask_path = os.path.join(mask_folder, mask_file)
            mask_img = np.array(Image.open(mask_path), order="F")
            mask_polygon = polygonFromMask(mask_img)
            #print(mask_polygon)
            mask_rle = encode(mask_img) 
            bbox = toBbox(mask_rle)
            annotations.append({
                "bbox": bbox.tolist(),
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
                "segmentation": [mask_polygon],
            })
        with open(os.path.join(train_set_path, img_folder, "polygon.pkl"), "wb") as f:
            pickle.dump(annotations, f)

if __name__ == "__main__":
    preprocess_mask()