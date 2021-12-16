import os
import pickle
import json
from detectron2.data import MetadataCatalog, DatasetCatalog

def nucleus_dataset(dataset_path="/home/nemo/VRDL2021/NYCU_VRDL_HW3/dataset", mode="train", mask_type="rle"):
    """
    Returns a list of dict, representing each training image
    """
    train_set = []
    train_set_path = os.path.join(dataset_path,'train')
    for img_folder in os.listdir(train_set_path):
        file_name = os.path.join(train_set_path, img_folder, "images", img_folder + ".png")
        height, width = 1000, 1000
        image_id = img_folder
        if mask_type=="rle":
            with open(os.path.join(train_set_path, img_folder, "masks.pkl"), "rb") as f:
                annotations = pickle.load(f)
        else:
            assert mask_type=="polygon", "mask_type must be either 'rle' or 'polygon'"
            with open(os.path.join(train_set_path, img_folder, "polygon.pkl"), "rb") as f:
                annotations = pickle.load(f)
        img_dict = {
            "file_name": file_name,
            "height": height,
            "width": width,
            "image_id": image_id,
            "annotations": annotations
        }
        train_set.append(img_dict)
    if mode == "train":
        return train_set[:int(len(train_set))]
    elif mode == "validate":
        return train_set[int(len(train_set)):]
    else:
        assert mode=="test", "mode must be either train, validate or test"
        with open(os.path.join(dataset_path,"test_img_ids.json"), "r") as f:
            test_json = json.load(f)
        test_set = []        
        for test_img in test_json:
            file_name = os.path.join(dataset_path,"test", test_img["file_name"])
            height , width = test_img["height"], test_img["width"]
            image_id = test_img["id"]
            img_dict = {
                "file_name": file_name,
                "height": height,
                "width": width,
                "image_id": image_id
            }
            test_set.append(img_dict)   
        return test_set

DatasetCatalog.register("nucleus_train", lambda : nucleus_dataset(mode="train"))
MetadataCatalog.get("nucleus_train").set(thing_classes=["nucleus"]) # set the thing_classes
DatasetCatalog.register("nucleus_val", lambda : nucleus_dataset(mode="validate"))
MetadataCatalog.get("nucleus_val").set(thing_classes=["nucleus"]) # set the thing_classes
MetadataCatalog.get("nucleus_val").set(evaluator_type="coco") # set the evaluator_type
DatasetCatalog.register("nucleus_test", lambda : nucleus_dataset(mode="test"))


# test code
if __name__ == "__main__":
    train_set = nucleus_dataset()
    print(train_set[0])