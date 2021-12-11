import os
import pickle

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
    else:
        return train_set[int(len(train_set) * 0.8):]

# test code
if __name__ == "__main__":
    train_set = nucleus_dataset()
    print(train_set[0])