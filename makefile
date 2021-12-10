.PHONY: install preprocess_mask clean_mask all 

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