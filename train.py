import random
import numpy as np
import os
import cv2
import argparse
import json
from datetime import datetime
from pycocotools.mask import encode, toBbox
from nucleus_dataset import nucleus_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import LazyConfig
from tqdm import tqdm



nucleus_metadata = MetadataCatalog.get("nucleus_train")
nucleus_metadata = MetadataCatalog.get("nucleus_val")

"""

#Visualize Annotation (Check loaded correctly)

dataset_dicts = nucleus_dataset()
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    print(type(d["annotations"][0]["bbox"]))
    visualizer = Visualizer(img[:, :, ::-1], metadata=nucleus_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("Nucleus", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
"""

parser = argparse.ArgumentParser(description="Detectron2 Training")
parser.add_argument("--resume", default=False, action="store_true", help="Resume from the model")
parser.add_argument("--outdir", type=str, default="default", help="Output directory")
parser.add_argument("--eval", default=False, action="store_true", help="Evaluate the model")
parser.add_argument("--model_path", type=str, default="", help="Path to the model")
parser.add_argument("--config_path", type=str, default="", help="Path to the config")
parser.add_argument("--visualize", default=False, action="store_true", help="Visualize Evaluation Results")

cmd_args = parser.parse_args()

cfg = get_cfg()
if not cmd_args.resume and not cmd_args.eval:
    if cmd_args.outdir == "default":
        cfg.OUTPUT_DIR = f'output/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    else:
        cfg.OUTPUT_DIR = cmd_args.outdir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
else:
    cfg.OUTPUT_DIR = cmd_args.outdir


cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
#cfg = LazyConfig.load("detectron2_configs/new_baselines/mask_rcnn_R_101_FPN_100ep_LSJ.py")

cfg.DATASETS.TRAIN = ("nucleus_train",)
cfg.DATASETS.TEST = ("nucleus_val",)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.SOLVER.IMS_PER_BATCH = 1 # originally 16
cfg.SOLVER.BASE_LR = 0.00125 # originally 0.02
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = (2000, 2750)
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512 # original is 256
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.MASK_FORMAT = "bitmask"  # alternative: "polygon"
cfg.INPUT.MIN_SIZE_TRAIN = (512, 512)
cfg.INPUT.RANDOM_FLIP = "horizontal"
cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.TYPE = "relative_range"
cfg.INPUT.CROP.SIZE = (0.5, 0.5) # optionally (0.7, 0.7) or (0.2, 0.2) or (0.3, 0.3)
cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28 # originally 14
cfg.SOLVER.CHECKPOINT_PERIOD = 250 # save checkpoint every 250 iterations
#print(cfg.model)

#cfg = LazyConfig.load(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x.py"))

if cmd_args.eval == False:
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=cmd_args.resume)
    trainer.train()
else:
    cfg.MODEL.WEIGHTS = os.path.join(cmd_args.model_path)  # path to the model we just trained
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    # Load and display evaluation results
    dataset_dicts = nucleus_dataset(mode="train")
    if cmd_args.visualize:
        for d in dataset_dicts[:3]:    
            img = cv2.imread(d["file_name"])
            outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            
            v = Visualizer(img[:, :, ::-1],
                    metadata=nucleus_metadata, 
                    scale=1.0, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )

            visualizer = Visualizer(img[:, :, ::-1], metadata=nucleus_metadata, scale=1.0 )
            vis = visualizer.draw_dataset_dict(d)
            gt_img = vis.get_image()[:, :, ::-1]
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out_img = out.get_image()[:, :, ::-1]
            cv2.imshow("Eval_Result",np.concatenate((gt_img, out_img), axis=1))
            cv2.waitKey(0)
        exit(0)
    else:
        # output  COCO style evaluation results
        """
        [{
            "image_id": int, 
            "category_id": int, 
            "segmentation": RLE, 
            "score": float,
        }]
        """
        coco_result = []
        dataset_dicts = nucleus_dataset(mode="test")
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

            image_id = d["image_id"]
            category_id = 1
            for i in tqdm(range(outputs["instances"].scores.shape[0])):
                segmentation = encode(np.array(outputs["instances"].pred_masks[i].cpu(), order="F"))
                segmentation["counts"] = str(segmentation["counts"],'utf-8')
                score = outputs["instances"].scores[i].item()
                coco_result.append({"image_id": image_id, "category_id": category_id, "segmentation": segmentation, "score": score})
        with open(os.path.join(cfg.OUTPUT_DIR, "answer.json"), "w") as f:
            json.dump(coco_result, f)




