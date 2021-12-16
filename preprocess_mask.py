import os
import numpy as np
from PIL import Image
from pycocotools.mask import encode, toBbox
import tqdm
import pickle
from detectron2.structures import BoxMode


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
            mask_rle = encode(mask_img) 
            bbox = toBbox(mask_rle)
            annotations.append({
                "bbox": bbox.tolist(),
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
                "segmentation": mask_rle,
            })
        with open(os.path.join(train_set_path, img_folder, "masks.pkl"), "wb") as f:
            pickle.dump(annotations, f)

if __name__ == "__main__":
    preprocess_mask()