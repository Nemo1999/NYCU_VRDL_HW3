# NYCU_VRDL_HW3
This is the repo for HW3 of Vision Recognition with Deep Learing in NYCU

We do instance segmentation on [nucleus dataset](https://www.kaggle.com/c/data-science-bowl-2018/data)

## Ground Truth vs Prediction 


![training result visualized](https://i.imgur.com/ohw8uKT.jpg)

--- 
![training result visualized](https://i.imgur.com/BJafMSk.jpg)


## References
- [MASK RCNN](https://arxiv.org/abs/1703.06870)
- [detectron2](https://github.com/facebookresearch/detectron2)

## Requirements
- detectron2 (follow [Installation Guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), install opencv to enable visualization)
- python >= 3.6
- pytorch >= 1.8
- make
- unzip
- opencv (optional, only for visualization)

## Setup
1. Install dependencies
```bash
make install
```

2. Download dataset and process the mask files
``` bash
make getdataset
make preprocess_mask
```

## Train
```bash
make train
```
Checkpoints and training log will be saved in `output/` folder 

## Reproduce 
```bash
make reproduce
```
The result will be saved in `output/run_name/answer.json`

## Custom Usage

```
python train.py [options]
```
#### Options: 
- `--resume` resume training 
- `--outdir "your_path"` specify output directory
- `--eval` evaluation mode (not training), default behaviour will save `answer.json` to `--outdir`
- `--model_path "your_path"` specify path to checkpoint model for evaluation (only take effect when `--eval` is set)
- `--visualize` visualize the evaluation result instead of saving json file (only take effect when `--eval` is set)
