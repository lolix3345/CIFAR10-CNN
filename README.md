# CIFAR10-CNN
A CNN model and the script for training it on the CIFAR-10 dataset

## Requirements

This program was developed on Python 3.8.16 with : 
|Module|Version|
|----|----|
|Pytorch|2.0.1|
|Torchvision|0.15.2|

The cuda version used was 11.8

The following modules are also needed :
- tensorboard
- ptflops
- numpy

## Quickstart

You first need to locate the directory where the batch files of the CIFAR-10 dataset are. Once you did this, you can launch the main.py script as follows


```
python main.py --dataset-dir "Path to the dataset" --batch-size 128 --model-width 16 --n-epochs 10 --testing-interval 2 --use-gpu
```

Informations about the other possible arguments are present in main.py.
