import random
import os
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from nucleus_dataset import nucleus_dataset
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import LazyConfig

DatasetCatalog.register("nucleus_train", lambda : nucleus_dataset(mode="train"))
MetadataCatalog.get("nucleus_train").set(thing_classes=["nucleus"]) # set the thing_classes
DatasetCatalog.register("nucleus_val", lambda : nucleus_dataset(mode="validation"))
MetadataCatalog.get("nucleus_val").set(thing_classes=["nucleus"]) # set the thing_classes
nucleus_metadata = MetadataCatalog.get("nucleus_train")
nucleus_metadata = MetadataCatalog.get("nucleus_val")
"""
Visualize Annotation (Check loaded correctly)

dataset_dicts = nucleus_dataset()
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    print(type(d["annotations"][0]["bbox"]))
    visualizer = Visualizer(img[:, :, ::-1], metadata=nucleus_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("Nucleus", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
"""




cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x.py"))
cfg.OUTPUT_DIR = "./output"
cfg.DATASETS.TRAIN = ("nucleus_train",)
cfg.DATASETS.TEST = ("nuclues_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "Pretrained/model_final_f1362d.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.MASK_FORMAT = "bitmask"  # alternative: "polygon"
#cfg = LazyConfig.load(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x.py"))

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()




