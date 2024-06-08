# NN-Compression

## Requirements
- pytorch
- torchvision
- pytorch lighting
- wandb
- numpy
- scipy
- tqdm
- PIL
- https://github.com/alibaba/TinyNeuralNetwork

## Dataset
The dataset used for this project is the Cityscapes Image Pairs dataset available at https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs. After the the dataset is downloaded, simply extract the dataset and then place it in `data` directory in the root directory of this repository.

## Model
### Task 
In this project we train a semantic segmentation model
### Architecture
The model is of the UNet architecture. It is a combination of EfficientNetB0 as the encoder and FFNet as the decoder.
### Training model from scratch
To train the model from scratch simply run:
```
python train.py
``` 
### Trained model
A trained model checkpoint is also provided - [model.ckpt](model.ckpt)

## Model quantization
Model quantization is performed and evaluated in [compression.ipynb](compression.ipynb). The quantized model is also provided in the [quant_output](quant_output) directory.