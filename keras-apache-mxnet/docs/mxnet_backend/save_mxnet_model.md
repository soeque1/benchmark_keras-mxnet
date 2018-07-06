# Save MXNet model from Keras-MXNet

## Table of Contents

1. [Objective](#objective)
2. [Train and Save a Convolutional Neural Network (CNN) Model for MNIST Dataset](#train-and-save-a-convolutional-neural-network-cnn-model-for-mnist-dataset)
3. [Import the model in MXNet for Inference](#import-the-model-in-mxnet-for-inference)
4. [References](#references)

## Objective

In this tutorial, we show how to train a model in Keras-MXNet, export the trained model as Apache MXNet model using `keras.models.save_mxnet_model()` API, and use MXNet natively for inference.

The Keras interface is known for its easy to use APIs, enabling fast prototyping in deep learning research. 
MXNet is known for its high performance, production ready engine. With Keras-MXNet, you get an out-of-the-box API to 
export most trained Keras models in MXNet model format. 
You can now use Keras-MXNet for training the model and MXNet for inference in production. You can use `keras.models.save_mxnet_model()` API to save 
the models trained on the CPU, a single GPU or multiple GPUs.

You can use any language bindings supported by MXNet (Scala/Python/Julia/C++/R/Perl) for performing inference with these models!

`Warning` Not all Keras operators and functionalities are supported with MXNet backend. For more information, view the the list
 of known issues and unsupported functionalities [here](https://github.com/awslabs/keras-apache-mxnet/issues/18).

This API accepts the following arguments:
* model: Keras model instance to be saved as MXNet model.
* prefix: Prefix name of the saved Model (symbol and params) files. Model will be saved as 'prefix-symbol.json' and 'prefix-epoch.params'.
* epoch: (Optional) Tag the params file with epoch of the model being saved. Default is 0. Model params file is saved as 'prefix-epoch.params' or 'prefix-0000.params' by default.


To summarize, all you have to do is to call the `keras.models.save_mxnet_model()` API by passing the trained Keras 
model to be exported in MXNet model format.

```python
# ... Assuming you have built and trained a Model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
          
# Save the trained Keras Model as MXNet Model
keras.models.save_mxnet_model(model=model, prefix='my_model')

# You get the MXNet model - (my_model-symbol.json, my_model-0000.params) in your current directory.
# Symbol and Params are two files representing a native MXNet model.

```
 
## Train and save a Convolutional Neural Network (CNN) model for MNIST dataset

We provide the following example for building a simple CNN model in Keras for [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit recognition dataset. As you follow the example, you will save the model in the 
MXNet model format. You will use the `keras.models.save_mxnet_model()` API.

```python
# Reference - https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/mnist_cnn.py

# Step 1: Train a CNN model for MNIST dataset.

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Import the save_mxnet_model API
from keras.models import save_mxnet_model

batch_size = 128
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Step 2: Save the model in MXNet model format.
# data_names and data_shapes are values of the parameters to be used when loading the Model in MXNet.
data_names, data_shapes = save_mxnet_model(model=model, prefix='mnist_cnn', epoch=0)
``` 

After running this script, you will now have two files for an MXNet model in the current directory.
* mnist_cnn-symbol.json
* mnist_cnn-0000.params

In the next section, we show how to load this model in the native MXNet engine and perform inference.

## Import the model in MXNet for Inference

The `keras.model.save_mxnet_model()` API will return the `data_names` and `data_shapes` to be used for binding the model with MXNet engine. 

```python
import numpy as np
import mxnet as mx

# Step1: Load the model in MXNet

# Use the same prefix and epoch parameters we used in save_mxnet_model API.
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix='mnist_cnn', epoch=0)

# We use the data_names and data_shapes returned by save_mxnet_model API.
mod = mx.mod.Module(symbol=sym, 
                    data_names=['/conv2d_1_input1'], 
                    context=mx.cpu(), 
                    label_names=None)
mod.bind(for_training=False, 
         data_shapes=[('/conv2d_1_input1', (1,1,28,28))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

# Step2: Perform inference

# Now, the model is loaded in MXNet and ready for Inference!
# We load MNIST dataset and demonstrate inference.
mnist = mx.test_utils.get_mnist()
labels = mnist['test_label']
test_data = mnist['test_data']
data_iter = mx.io.NDArrayIter(test_data, None, 1)
result = mod.predict(data_iter)

# Check what is the predicted value and actual value
# We have predicted 10000 samples in test_data. Use different indexes to see different sample results.
idx = 1020
print("Predicted - ", np.argmax(result[idx].asnumpy()))
print("Actual - ", labels[idx])
```

That's it! We trained a CNN model with Keras interface and used MXNet native engine in Python for inference. Also 
note that we can use any language binding supported by MXNet (Scala/Python/Julia/C++/R/Perl) for inference based on your 
production environment setup and requirements.

## References
1. [MXNet Module](https://mxnet.incubator.apache.org/api/python/module/module.html)
2. [MXNet Predicting with Pre-Trained Model](https://mxnet.incubator.apache.org/tutorials/python/predict_image.html)
3. [Keras MNIST CNN code](https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/mnist_cnn.py)
