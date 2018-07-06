# Performance Tuning - Keras with MXNet Backend

## Table of Contents
1. [Objective](#objective)
2. [Channels First Image Data Format for CNN](#channels-first-image-data-format-for-cnn)
3. [Optimized MXNet binaries](#optimized-mxnet-binaries)
4. [Using MXNet Profiler](#using-mxnet-profiler)
5. [References](#references)

## Objective
In this document, we will discuss various tips and techniques to improve training/inference performance in Keras with MXNet backend.

## Channels First Image Data Format for CNN

When you install the `keras-mxnet`, by default, the following values are set.

```json
backend: mxnet
image_data_format: channels_last
```

We strongly recommend changing the image_data_format to `channels_first`. MXNet is significantly faster on 
`channels_first` data. `channels_first` data format is optimal for training on NVIDIA GPUs with cuDNN, see 
the [TensorFlow Performance Guide](https://www.tensorflow.org/performance/performance_guide#data_formats) for more 
details.

The default is set to `channels_last` with an objective to be compatible with the majority of existing users of 
Keras-TensorFlow.

You will see the following user warning with `channels_last` data format in MXNet backend.

```
UserWarning: MXNet Backend performs best with `channels_first` format. Using `channels_last` will significantly reduce performance due to the Transpose operations. For performance improvement, please use this API`keras.utils.to_channels_first(x_input)`to transform `channels_last` data to `channels_first` format and also please change the `image_data_format` in `keras.json` to `channels_first`.Note: `x_input` is a Numpy tensor or a list of Numpy tensor
```

Follow the below steps for moving to high performance 'channels_first' data format.

### Step 1. Update the Keras config file

Update the `image_data_format` setting to `channels_first` in the [Keras config file](https://keras.io/backend/#kerasjson-details), `keras.json`. You can find the Keras config file in your home directory `~/.keras/keras.json`.

```json
"image_data_format": "channels_first", 
"backend": "mxnet"
```

### Step 2. Update data to channels first format

If your data is in the channels last format (Ex: shape=(256, 256, 3)), you need to convert the data to channels first format (Ex: shape=(3, 256, 256)). 

Keras-MXNet has a utility API [`keras.utils.to_channels_first()`](https://github.com/awslabs/keras-apache-mxnet/blob/master/keras/utils/np_utils.py#L55) that can take a single input or a batch of data in channels last format and return back the data in channels first format.


```
NOTE:
    Keras dataset API is not consistent across different dataset. It returns a few datasets based on the 
    image_data_format in keras.json (Ex: CIFAR10 dataset] and a few datasets always in the channels_last format (Ex: 
    MNIST dataset). It is recommended to always verify the shape of data returned.
```

### Step 3. Update the input_shape argument when constructing the model

You need to update the input shapes in the network to channels first format. For example, using Conv2D on the color 
images with size 256 by 256, you need to set:

For channels last format, set the `input_shape` parameter as follows:

```
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256,256,3)))
```

For channels first format, set the `input_shape` parameter as follows:

```
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, 256,256)))
```

```
NOTE:
    It is recommended practice to pass the input_shape based on the shape field of your input tensor. For example:
    
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

```

## Optimized MXNet binaries

MXNet PyPi packages are distributed in multiple flavors, where each optimized for a specific underlying hardware. 
Please install the correct MXNet flavor optimized for your instance type for the best performance.

1. Default: `pip install mxnet`
 
2. Optimized for Intel CPUs with MKL: `pip install mxnet-mkl`

3. Optimized for NVIDIA GPUs with CUDA9: `pip install mxnet-cu90` 

4. Optimized for NVIDIA GPUs and Intel CPUs: `pip install mxnet-cu90-mkl`

5. Refer to the [MXNet installation guide](https://mxnet.incubator.apache.org/install/index.html) for more 
configurations to build from the source.

Please see the [MXNet Performance Guide](https://mxnet.incubator.apache.org/faq/perf.html) for more detailed 
performance tips.

## Using MXNet Profiler

MXNet has a built-in profiler that provides detailed information about the execution times for each operator. The MXNet 
profiler complements general profiling tools like `nvprof` and `gprof` by summarizing execution time at the operator 
level, rather than at a function, kernel, or instruction level. Review [MXNet Performance Guide](https://mxnet.incubator.apache.org/faq/perf.html) for more details.

## References
1. [MXNet Performance Guide](https://mxnet.incubator.apache.org/faq/perf.html)
2. [TensorFlow Performance Guide](https://www.tensorflow.org/performance/performance_guide#data_formats)

