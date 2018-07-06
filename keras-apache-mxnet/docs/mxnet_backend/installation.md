# Installation

## Table of Contents

1. [Install Keras with Apache MXNet backend](#1-install-keras-with-apache-mxnet-backend)
    1. [CPU Setup](#11-cpu-setup)
    2. [GPU Setup](#12-gpu-setup)
2. [Configure Keras backend and image_data_format](#2-configure-keras-backend-and-image_data_format)
3. [Validate the Installation](#3-validate-the-installation)
4. [Train a simple handwritten digit recognition model](#4-train-a-simple-handwritten-digit-recognition-model)
5. [Next Steps](#5-next-steps) 

## 1 Install Keras with Apache MXNet backend

`Warning` Not all Keras operators and functionalities are supported with MXNet backend. For more information, view the the list
 of known issues and unsupported functionalities [here](https://github.com/awslabs/keras-apache-mxnet/issues/18).

Steps involved
1. Install optional Keras dependencies
* CUDA and cuDNN (required if you plan on running Keras on GPU).
* HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (required if you plan on saving Keras models to disk).
* graphviz and pydot (used by visualization utilities to plot model graphs).

2. Install MXNet - [https://github.com/apache/incubator-mxnet](https://github.com/apache/incubator-mxnet)
3. Install Keras with MXNet backend - [https://github.com/awslabs/keras-apache-mxnet](https://github.com/awslabs/keras-apache-mxnet)

```
NOTE 

The following installation instructions are tested on Ubuntu 14.04/16.04 and Mac OS EL Capitan and Sierra.
```

### 1.1 CPU Setup

#### Install optional dependencies

```bash
    # install python and pip if not already installed
    $ sudo apt-get update
    $ sudo apt-get install -y wget python
    $ wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py
    # install optional dependencies
    $ pip install h5py --user
    $ pip install graphviz --user
    $ pip install pydot --user
```
#### Install MXNet

```bash
    $ pip install mxnet --user
```
If you would like to use MXNet with MKL for high performance on CPU, install `mxnet-mkl` with below command.
```bash
    $ pip install mxnet-mkl --user
```

#### Install Keras with MXNet backend

```bash
    $ pip install keras-mxnet --user
```

### 1.2 GPU Setup

#### Install dependencies

```bash
    # install python and pip if not already installed
    $ sudo apt-get update
    $ sudo apt-get install -y wget python
    $ wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py
    # install optional dependencies
    $ pip install h5py --user
    $ pip install graphviz --user
    $ pip install pydot --user
```

#### Install CUDA and cuDNN

```
NOTE

If you are running on AWS, you can launch an EC2 instance with AWS Deep Learning Base AMI to get CUDA, cuDNN and other required dependencies pre-configured. Follow the [tutorial here](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-base.html)

```
Install the following NVIDIA libraries to setup with GPU support:

1. Install CUDA 9.1 following the NVIDIA's [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
2. Install cuDNN 7.1 for CUDA 9.1 following the NVIDIA's [installation guide](https://developer.nvidia.com/cudnn). You may need to register with NVIDIA for downloading the cuDNN library.

```
NOTE 

Make sure to add the CUDA install path to `LD_LIBRARY_PATH`.
```
For example, on an Ubuntu machine, if you have downloaded CUDA debian package (`cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb`) and cuDNN 7.1 library (`cudnn-9.1-linux-x64-v7.1.tgz`), below are set of commands you run to setup CUDA and cuDNN.

```bash

#  Setup CUDA 9.0.
$  sudo apt-get update
$  sudo apt-get install build-essential
$  sudo apt-get install linux-headers-$(uname -r)
#  Assuming you have downloaded CUDA deb package from https://developer.nvidia.com/cuda-downloads
$  sudo dpkg -i cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb
$  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
$  sudo apt-get update
$  sudo apt-get install cuda

$  export CUDA_HOME=/usr/local/cuda
$  export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
$  export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/:$LD_LIBRARY_PATH

#  Setup cuDNN 7.1 for CUDA 9.1
#  Assuming you have registered with NVIDA and downloaded cuDNN 7.1 for CUDA 9.1 from
#  https://developer.nvidia.com/rdp/cudnn-download
$  tar -xvzf cudnn-9.1-linux-x64-v7.1.tgz
$  sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
#  Use cp -a to copy symlink and avoid copying duplicated files 
$  sudo cp -a cuda/lib64/* /usr/local/cuda/lib64/
$  sudo chmod a+r /usr/local/cuda/include/cudnn.h
$  /usr/local/cuda/lib64/libcudnn*
```

You can verify your CUDA setup with the following commands.

```bash
$  nvcc --version
$  nvidia-smi
```

#### Install MXNet

```bash
    # If you use CUDA9. If CUDA8, use, mxnet-cu80
    $ pip install mxnet-cu90
```

#### Install Keras with MXNet backend

```bash
    $ pip install keras-mxnet --user
```

## 2 Configure Keras backend and image_data_format

When you install the `keras-mxnet`, by default, the following values are set.

```json
backend: mxnet
image_data_format: channels_last
```

We strongly recommend changing the image_data_format to `channels_first`. MXNet is significantly faster on 'channels_first' data. Default is set to 'channels_last' with an objective to be compatible with majority of existing users of Keras. See [performance tips guide](performance_guide.md) for more details.

```
NOTE

If you cannot find ~/.keras/keras.json file, just load the Keras library once to get the config state created. 

    $ python
    >>> import keras as k
        Using mxnet backend
```
## 3 Validate the Installation

You can validate the installation by trying to import Keras in Python terminal and verifying that Keras is using *mxnet* backend.

```bash
    $ python
    >>> import keras as k
        Using mxnet backend
```
Next, get hands-on by training a simple Multi Layer Perceptron (MLP) model for handwritten digit recognition using MNIST dataset.

## 4 Train a simple handwritten digit recognition model

In this section you will verify the Keras-MXNet installation and try out training a simple Multi-Layer Perceptron (MLP) model.

[awslabs/keras-apache-mxnet/examples](https://github.com/awslabs/keras-apache-mxnet/tree/master/examples) already consists of code to train the model. For simplicity, we only submit the model training job as a black box to see the training in action and do not focus on teaching what the model does.

```bash
    # Clone the awslabs/keras-apache-mxnet repository.
    $ git clone --recursive https://github.com/awslabs/keras-apache-mxnet

    # Launch the model training job.
    $ python keras-apache-mxnet/examples/mnist_mlp.py
```
Your output should look something like below.

```

    Using MXNet backend.
    60000 train samples
    10000 test samples
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    dense_1 (Dense)                  (None, 512)           401920      dense_input_1[0][0]              
    ____________________________________________________________________________________________________
    activation_1 (Activation)        (None, 512)           0           dense_1[0][0]                    
    ____________________________________________________________________________________________________
    dropout_1 (Dropout)              (None, 512)           0           activation_1[0][0]               
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 512)           262656      dropout_1[0][0]                  
    ____________________________________________________________________________________________________
    activation_2 (Activation)        (None, 512)           0           dense_2[0][0]                    
    ____________________________________________________________________________________________________
    dropout_2 (Dropout)              (None, 512)           0           activation_2[0][0]               
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 10)            5130        dropout_2[0][0]                  
    ____________________________________________________________________________________________________
    activation_3 (Activation)        (None, 10)            0           dense_3[0][0]                    
    ====================================================================================================
    Total params: 669,706
    Trainable params: 669,706
    Non-trainable params: 0
    ____________________________________________________________________________________________________
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    /home/ubuntu/.local/lib/python2.7/site-packages/mxnet/module/bucketing_module.py:368: UserWarning: Optimizer created manually outside Module but rescale_grad is not normalized to 1.0/batch_size/num_workers (1.0 vs. 0.0078125). Is this intended?
      force_init=force_init)
    60000/60000 [==============================] - 98s - loss: 1.2175 - acc: 0.6823 - val_loss: 0.5459 - val_acc: 0.8675
    Epoch 2/20
    43136/60000 [====================>.........] - ETA: 25s - loss: 0.5554 - acc: 0.8458 
```

Congratulations! You have successfully installed Apache MXNet, Keras with MXNet backend and trained your first model!

## 5 Next Steps

* Read the Keras documentation at [Keras.io](https://keras.io/).
* For more examples explore [keras/examples](https://github.com/awslabs/keras-apache-mxnet/tree/master/examples) directory.
* Tutorial on how to use multiple GPUs with MXNet backend - [Multi-GPU Distributed Training with Keras and Apache MXNet](multi_gpu_training.md)
* Tutorial on how to save MXNet model from Keras-MXNet - [Save MXNet model from Keras-MXNet](save_mxnet_model.md)
* Tips for high performance with Keras-MXNet - [Performance Guide for MXNet Backend](performance_guide.md)
