# CenterPillarNet
An anchor free method for pointcloud object detecion.

[![ros passing](https://img.shields.io/badge/ros-passing-brightgreen.svg)](https://github.com/wangx1996/CenterPillarNet)  [![torch 1.6](https://img.shields.io/badge/torch-1.6-red.svg)](https://github.com/wangx1996/CenterPillarNet)  [![python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://github.com/wangx1996/CenterPillarNet)

### Result
![Image text](https://github.com/wangx1996/CenterPillarNet/blob/main/img/pillar_img.gif)


### Introdcution

This is an anchor free method for pointcloud object detecion. 

This project is not finished yet, it has a lot of parts to be improved. 

If you are intreseted in this project, you can try to change the code and make this work better.

If you have any idea on this work, please contact me.

More details I will put it on wiki.

### 1.Clone Code

    git clone https://github.com/maudzung/CenterPillarNet.git CenterPillarNet
    cd CenterPillarNet/
    
### 2.Install Dependence
#### 2.1 base pacakge
    pip install -r requirements.txt
    
for anaconda

    conda install scikit-image scipy numba pillow matplotlib
    pip install fire tensorboardX protobuf opencv-python

#### 2.2 spconv
First download the code

    git clone https://github.com/traveller59/spconv.git --recursive spconv
    cd spconv
    
Build the code
    
    python setup.py bdist_wheel
    cd ./dist
    pip install ***.whl
    
#### 2.3 DCN

Please download DCNV2 from [https://github.com/jinfagang/DCNv2_latest](https://github.com/jinfagang/DCNv2_latest) to fit torch 1.

Put the file into 

    ./src/model/
    
then 

    ./make.sh
    
#### 2.4 Setup cuda for numba

    export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
    export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
    export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
 

### 3. Prepaer data

KITTI dataset

You can Download the KITTI 3D object detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

It includes:
Velodyne point clouds (29 GB)

Training labels of object data set (5 MB)

Camera calibration matrices of object data set (16 MB)

Left color images of object data set (12 GB) 

Data structure like

    └── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   └── velodyne
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           └── velodyne
### 4. How to Use

First, make sure the dataset dir is right in your train.py file

Then run

    python train.py --gpu_idx 0 --arch dla_34 --saved_fn cpdla --batch_size 1
    
Tensorboard
    
    cd logs/<saved_fn>/tensorboard/
    tensorboard --logdir=./
    
Actually, I only have one RTX2070, so the batch_size must be one, but if you have morce GPUs, you can try other number of batchsize.

if you want to test the work

    python test.py --gpu_idx 0 --arch dla_34 --pretrained_paht ../checkpoints/**/**
    
if you want to evaluate the work

    python evaluate.py --gpu_idx 0 --arch dla_34 --pretrained_paht ../checkpoints/**/**
    
also you can choose another method to evaluate the work:

first you need to run 

    python evaluatefiles.py --gpu_idx 0 --arch dla_34 --pretrained_paht ../checkpoints/**/**

then you can use this [project](https://github.com/traveller59/kitti-object-eval-python) to eval.


### Reference

Thanks for all the great works.

[1] [SFA3D](https://github.com/maudzung/SFA3D)

[2] [CenterNet: Objects as Points](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.07850), [[PyTorch Implementation](https://github.com/xingyizhou/CenterNet)]

[3] [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/pdf/1812.05784.pdf),[[PyTorch Implementation](https://github.com/traveller59/second.pytorch)]

[4] [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211) [[final version code](https://github.com/jinfagang/DCNv2_latest)]

Inspired by

[1] [AFDet: Anchor Free One Stage 3D Object Detection](https://arxiv.org/abs/2006.12671)


