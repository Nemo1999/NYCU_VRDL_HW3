.PHONY: install all 

install:
	@echo "Installing dependencies"
	pip install gdown

getdataset: 
	@echo "Downloading dataset from Google Drive"
	gdown https://drive.google.com/uc?id=1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG -O dataset.zip
	unzip dataset.zip

