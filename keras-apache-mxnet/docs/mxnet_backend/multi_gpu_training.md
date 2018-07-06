# Multi-GPU Model Training with Keras-MXNet

## Table of Contents

1. [Overview](#overview)
2. [Objective](#objective)
3. [Prerequisites](#prerequisites)
4. [Prepare the Data](#prepare-the-data)
5. [Build the Network](#build-the-network)
6. [Build Multi-GPU Model and Compile](#build-multi-gpu-model-and-compile)
7. [Train the Model](#train-the-model)
8. [References](#references)

## Overview

In this tutorial, you will use [Keras](https://keras.io/), with [Apache MXNet](https://mxnet.incubator.apache.org/) 
backend, on a multi-GPU machine, to train a Convolutional Neural Network (CNN) model on [CIFAR10 small images dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

MXNet backend makes large scale multi-GPU model training in Keras significantly faster! See [benchmark results](../../benchmark/README.md) for more details.

## Objective

The main objective of this tutorial is to show *how to use multiple GPUs for training the neural network using Keras with MXNet backend*.

MXNet backend supports Keras's [multi_gpu_model](https://keras.io/utils/#multi_gpu_model) API for distributed multi-gpu model training. All you have to do is pass either a list of GPU IDs or the number of GPUs to be used for training.

For example, when training with four GPUs, you configure the training by passing `gpus=4`.

```python
model = Sequential()
... Build the model ...

# Build multi_gpu_model
model = keras.utils.multi_gpu_model(model, gpus=4)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
```


That's it! MXNet backend will use 4 GPUs for training your model!

Below is a more detailed tutorial to train a Convolutional Neural Network (CNN) model on [CIFAR10 small images dataset](https://www.cs.toronto.edu/~kriz/cifar.html) using Keras with MXNet backend.

```
Note:
    You cannot pass gpus=1. By default, on a GPU machine, MXNet backend uses the first GPU device.

```
## Prerequisites

1. GPU machine with CUDA and cuDNN installed
2. Keras
3. MXNet with GPU support

Follow the step by [step installation instructions](installation.md#12-gpu-setup) to set up your machine with 
Keras-MXNet.

## Prepare the Data

CIFAR10 is a dataset of 50,000 32x32 color (3 channels) training images, labeled over 10 categories, and 10,000 test 
images. Load the CIFAR10 dataset using Keras's [*dataset*](https://keras.io/datasets/#cifar10-small-image-classification) utility.

We will use [*categorical cross entropy*](https://keras.io/losses/#categorical_crossentropy) to calculate the loss in model training. Hence, convert the integer representation of 10 categories, in the train and test dataset, to binary representation using [*to_categorical*](https://keras.io/utils/#to_categorical) function. 

```python
import keras
from keras.datasets import cifar10

num_classes = 10

# The data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
Y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
```

## Build the Network

Build a sequential model with 3 layers (1 input layer, 1 hidden layer, and 1 output layer). We do not dive deep in to the architectural details of the neural network. Our objective of this tutorial is to showcase how to use multiple GPUs in Keras with MXNet backend.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```
## Build Multi-GPU Model and Compile

You can easily use multiple GPUs for training with Keras's [multi_gpu_model](https://keras.io/utils/#multi_gpu_model) API. You can pass a list of GPU IDs (Ex: gpus=[0,1,2,3]) or just pass number of GPUs to use (Ex: gpus=4). 

```python
# Lets train on 4 GPUs
model = keras.utils.multi_gpu_model(model, gpus=4)

# Initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
  
# Let's train the model using RMSprop. Specify context with list of GPU IDs to be used during training.
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
```

## Train the Model

Most deep learning models dealing with images requires some form of image augmentation (modifying the image) techniques on training data for better accuracy, convergence and various other advantages for a good training process. Keras exposes a powerful collection of image augmentation techniques via *[ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator)* class. *ImageDataGenerator* augments the image during the training process i.e., it performs just-in-time image augmentation and feed the augmented image to the network.

We first create 'datagen' object of type *ImageDataGenerator* by specifying a set of image augmentations to perform on CIFAR training data images. Example - width_shift, height_shift, random_flip.

Data generator is then *fit* on to the training data followed by using *flow()* function of *ImageDataGenerator* to iterate over training data in batches for the given *batch_size* during model training process.

Common best practice is to use a batch_size of 32 per GPU. Since we are using 4 GPUs, we set batch_size to be 32*4.
```python
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32*4 # 32 per GPU. We use 4 GPUs in the example. Set batch_size to 32*4.
epochs = 50 # Increase this to 200 for higher accuracy.

# This will do preprocessing and realtime data augmentation:
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)

# Fit the model on the batches generated by datagen.flow().
history = model.fit_generator(datagen.flow(X_train, Y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=epochs,
                        validation_data=(X_test, Y_test))
```

## References
* This tutorial references code from [keras/examples/cifar10_cnn.py](https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/cifar10_cnn.py)
* Keras multi_gpu_model API - [https://keras.io/utils/#multi_gpu_model](https://keras.io/utils/#multi_gpu_model)
* Multi-GPU Training with Keras-MXNet [benchmark results](../../benchmark/README.md)
