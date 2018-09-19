# PyTorch Implementation of FusionGAN

This repository contains the code for implementation of the FusionGAN model described in the paper [**Generating a Fusion Image: One's Identity and Another's Shape**](https://arxiv.org/abs/1804.07455)

## Dependencies

The following are the dependencies required by this repository:

+ PyTorch v0.4
+ NumPy
+ SciPy
+ Pickle
+ PIL
+ Matplotlib

## Setup Instructions

First, download the repository on your local machine by either downloading it or running the following script in the terminal

``` Batchfile
git clone https://github.com/aarushgupta/FusionGAN.git
```
## Dataset Preparation

As the data is not publically available in the desired form, the frames of the required YouTube videos have been saved at [**this**](https://drive.google.com/drive/folders/1waOPQYOmQF1k0pT50uqp6STzYDdSv_5N?usp=sharing) Google Drive link. 

The link contains a compressed train folder which has the following 3 folders:
1. class1_cropped
2. class2_cropped
3. class3_cropped

Download the data from the link and put the folders according to the following directory structure:
```
/FusionGAN_root_directory/Dataset/train/class1_cropped
/FusionGAN_root_directory/Dataset/train/class2_cropped
/FusionGAN_root_directory/Dataset/train/class3_cropped
```

## Training Instructions
The hyperparameters of the model have been preset. To start training of the model, simply run the `train.py` file using the following command

``` cmd
python train.py
```
The code can also be run interactively using the Jupyter Notebooks `train.ipynb` or `FusionGAN.ipynb` provided in the repository.

## To-Do

1. [ ] Train the model and save checkpoints.
2. [ ] Add test script for the model.
3. [ ]  Add keypoint estimation for quantitative evaluation.
4. [ ] Remove the unused images in the dataset.