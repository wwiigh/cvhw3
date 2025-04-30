# NYCU Computer Vision 2025 Sprint HW3
Student ID: 313553002  
Name: 蔡琮偉
## Introduction
The objective of the homework is Instance Segmentation for colored medical images, use
 mask R-CNN as backbone. To finish the task, I apply Resnet 50 as backbone to Mask R
CNN, and with some data processing to get better performance. With the above method, the
 model get 0.40 on AP50. 
## How to install
### Install the environment
`
conda env create -f environment.yml
`  
If there are torch install error, please go to https://pytorch.org/get-started/locally/ to get correct torch version  
If get no tensorboard error in training, please run following command  
`
conda install -c conda-forge tensorboard
`  
or you can remove all SummaryWriter function  
### Pretained model and dataset download
Pretained model: [https://drive.google.com/file/d/1UEqBQEYE0ndyRfno_ySq5NYRD7iww57K/view?usp=drive_link](https://drive.google.com/file/d/1UEqBQEYE0ndyRfno_ySq5NYRD7iww57K/view?usp=drive_link)
dataset: [https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view?usp=drive_link](https://drive.google.com/file/d/1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI/view?usp=sharing)
### File structure
create model and data folder, put the model and unzip dataset to corresponding folder. It should look like this  
project-root/  
├── src/  
│   ├── dataset.py  
|   ├── eval.py  
|   ├── generate.py  
|   ├── model.py  
|   ├── test.py  
|   ├── train.py  
│   ├── utils.py  
├── data/       
│   ├── test_release/  
│   ├── train/  
│   └── test_image_name_to_ids.json  
├── model/  
│   ├── pretained.pth  
├── annotations.json       
├── README.md          
├── environment.yml    
└── .gitignore          
### How to use
For test, change the function test's parameter to the pretained model path then run  
`
python src/test.py
`  
For training, run  
`
python src/train.py
`  
You can change the model backbone in model.py, just uncommit self.model to ResNet or MobileNet version.
## Performance snapshot
![image](https://github.com/user-attachments/assets/6b0b8e9d-c4cd-4eef-91fc-fc18097e3c1d)



