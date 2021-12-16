.PHONY: install preprocess_mask clean_mask all train

install:
	@echo "Installing dependencies"
	pip install gdown
	pip install pillow
	pip install pycocotools
	pip install tqdm

preprocess_mask: dataset/train/TCGA-18-5592-01Z-00-DX1/masks.pkl
	@echo "Preprocessing masks"
	

clean_mask: 
	@echo "Cleaning masks"
	find dataset/train -name "masks.pkl" -delete


getdataset: 
	@echo "Downloading dataset from Google Drive"
	gdown https://drive.google.com/uc?id=1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG -O dataset.zip
	unzip dataset.zip

dataset/train/TCGA-18-5592-01Z-00-DX1/masks.pkl:
	python preprocess_mask.py

train: 
	@echo "Training model"
	python train.py --outdir output/masked_rcnn_r101_fpn_1x 

reproduce:
	@echo "Reproducing results..."
	mkdir pretrained_models
	@echo "Downloading model weights from google drive..." 
	gdown https://drive.google.com/uc?id=1_57UohNcPW0IyWYihwt3lLh3WVwZD4az -O pretrained_models/mask_rcnn_r101_fpn_1x.pth
	@echo "Inferencing on test set..."
	python train.py --model_path pretrained_models/mask_rcnn_r101_fpn_1x.pth --outdir output/masked_rcnn_r101_fpn_1x --eval 
	@echo "Result is saved in output/masked_rcnn_r101_fpn_1x/answer.json"
