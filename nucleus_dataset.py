import os
import pickle
import json

def nucleus_dataset(train_set_path="dataset/train", mode="train"):
    """
    Returns a list of dict, representing each training image
    """
    train_set = []
    for img_folder in os.listdir(train_set_path):
        file_name = os.path.join(train_set_path, img_folder, "images", img_folder + ".png")
        height, width = 1000, 1000
        image_id = img_folder
        with open(os.path.join(train_set_path, img_folder, "masks.pkl"), "rb") as f:
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
        return train_set[:int(len(train_set) * 0.8)]
    elif mode == "validate":
        return train_set[int(len(train_set) * 0.8):]
    else:
        assert mode=="test", "mode must be either train, validate or test"
        with open("dataset/test_img_ids.json", "r") as f:
            test_json = json.load(f)
        test_set = []        
        for test_img in test_json:
            file_name = os.path.join(train_set_path.replace("train", "test"), test_img["file_name"])
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
            

# test code
if __name__ == "__main__":
    train_set = nucleus_dataset()
    print(train_set[0])