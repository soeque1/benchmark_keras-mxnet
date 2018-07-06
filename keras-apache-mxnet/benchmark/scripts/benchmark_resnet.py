'''Trains a ResNet on the ImageNet/CIFAR10 dataset.
Credit:
Script modified from examples/cifar10_resnet.py
Reference:
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
'''

from __future__ import print_function

import argparse
import logging
import math
import os
import random
import time

import numpy as np
from logging_metrics import LoggingMetrics
from models.resnet import get_resnet_model
from models.timehistory import TimeHistory

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import multi_gpu_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    help='Dataset for training: cifar10 or imagenet')
parser.add_argument('--version',
                    help='Provide resnet version: 1 or 2')
parser.add_argument('--layers',
                    help='Provide number of layers: 20, 56 or 110')
parser.add_argument('--gpus',
                    help='Number of GPUs to use')
parser.add_argument('--train_mode',
                    help='Required for imagenet: train_on_batch or fit_generator')
parser.add_argument('--data_path',
                    help='Required for imagenet: path_to_imagenet_data')
parser.add_argument('--epoch', default=200, type=int,
                    help='Number of epoch')

args = parser.parse_args()

# Check args
if args.dataset not in ['cifar10', 'imagenet']:
    raise ValueError('Only support cifar10 or imagenet data set')

if args.version not in ['1', '2']:
    raise ValueError('Provide resnet version: 1 or 2')

if args.layers not in ['20', '56', '110']:
    raise ValueError('Provide number of layers: 20, 56 or 110')

if args.dataset == 'imagenet':
    if not args.train_mode or not args.data_path:
        raise ValueError('Need to provide training mode(train_on_batch or fit_generator) '
                         'and data path to imagenet dataset')

    if args.train_mode not in ['train_on_batch', 'fit_generator']:
        raise ValueError('Only support train_on_batch or fit_generator training mode')

if args.gpus is None or args.gpus < 1:
    num_gpus = 0
else:
    num_gpus = int(args.gpus)

# Training parameters
batch_size = 32 * num_gpus if num_gpus > 0 else 32
epochs = int(args.epoch)
num_classes = 1000 if args.dataset == 'imagenet' else 10
data_format = K._image_data_format
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# prepare logging
# file name: backend_data_format_dataset_model_batch_size_gpus.log
log_file = K.backend() + '_' + K.image_data_format() + '_' + args.dataset + '_resnet_v' + args.version + '_' + args.layers + '_batch_size' + str(batch_size) + '_' + str(num_gpus) + 'gpus'  # nopep8
logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler('{0}/{1}.log'.format('./', log_file))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
rootLogger.info('saving log file to {0}/{1}.log'.format('./', log_file))
rootLogger.info('using image format: %s' % data_format)

# Prepare Training Data
# CIFAR10 data set
if args.dataset == 'cifar10':
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    rootLogger.info('x_train shape: %s' % str(x_train.shape))
    rootLogger.info('%d train_samples' % x_train.shape[0])
    rootLogger.info('%d test samples' % x_test.shape[0])
    rootLogger.info('y_train shape: %s' % str(y_train.shape))

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

# ImageNet Dataset
if args.dataset == 'imagenet':
    input_shape = (256, 256, 3) if data_format == 'channels_last' else (3, 256, 256)
    if args.train_mode == 'fit_generator':
        train_datagen = ImageDataGenerator(
            rescale=1. / 255)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            args.data_path,
            target_size=(256, 256),
            batch_size=batch_size)
    else:
        # Load the imagenet data.
        train_images = []
        train_labels = []
        label_counter = 0
        for subdir, dirs, files in os.walk(args.data_path):
            for folder in dirs:
                for folder_subdir, folder_dirs, folder_files in \
                        os.walk(os.path.join(subdir, folder)):
                    for file in folder_files:
                        train_images.append(os.path.join(folder_subdir, file))
                        train_labels.append(label_counter)

                label_counter = label_counter + 1

        # shuffle data
        perm = list(range(len(train_images)))
        random.shuffle(perm)
        train_images = [train_images[index] for index in perm]
        train_labels = [train_labels[index] for index in perm]
        nice_n = math.floor(len(train_images) / batch_size) * batch_size


# process batch data for imagenet
def get_batch():
    index = 1

    global current_index

    B = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], input_shape[2]))
    L = np.zeros(shape=(batch_size))

    while index < batch_size:
        try:
            img = load_img(train_images[current_index].rstrip(),
                           target_size=(256, 256, 3))
            B[index] = img_to_array(img)
            B[index] /= 255

            L[index] = train_labels[current_index]
            index = index + 1
            current_index = current_index + 1
        except:
            rootLogger.info('Ignore image {}'.format(train_images[current_index]))
            current_index = current_index + 1

    return B, keras.utils.to_categorical(L, num_classes)


# Prepare Model
# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = int(args.version)

# Computed depth from supplied model parameter n
depth = int(args.layers)

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)


def lr_schedule(epoch):
    '''Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    '''
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


# get the model base on configs
model = get_resnet_model(version=version, input_shape=input_shape,
                         depth=depth, num_classes=num_classes)

# use multi gpu model for multi gpus
if num_gpus > 1:
    if K.backend() == 'mxnet':
        # MXNet merge weights on GPU by default
        model = multi_gpu_model(model, gpus=num_gpus)
    else:
        # merge weights on GPU
        model = multi_gpu_model(model, gpus=num_gpus, cpu_merge=False)

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
rootLogger.info('Training using: ' + model_type)

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'imagenet_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

time_callback = TimeHistory()
callbacks = [checkpoint, lr_reducer, lr_scheduler, time_callback]

# Run training, without data augmentation.
if args.dataset == 'imagenet':
    rootLogger.info('Not using data augmentation.')
    if args.train_mode == 'train_on_batch':
        for i in range(0, epochs):
            current_index = 0
            total_time = 0
            rootLogger.info('starting epoch {}/{}'.format(i, epochs))
            while current_index + batch_size < len(train_images):
                b, l = get_batch()
                # only record training time
                start_time = time.time()
                loss, accuracy = model.train_on_batch(b, l)
                end_time = time.time()
                total_time += 1000 * (end_time - start_time)
                batch_time = 1000 * (end_time - start_time)
                speed = batch_size * 1000.0 / batch_time if batch_time != 0 else 0
                rootLogger.info('batch {}/{} loss: {} accuracy: {} '
                                'time: {}ms speed: {}'.format(int(current_index / batch_size),
                                                              int(nice_n / batch_size), loss, accuracy,
                                                              batch_time, speed))

            rootLogger.info('finish epoch {}/{}  total epoch time: {}ms'.format(i, epochs, total_time))

    else:
        model.fit_generator(train_generator, epochs=epochs)
else:
    history_callback = model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(x_test, y_test),
                                 shuffle=True,
                                 verbose=0,
                                 callbacks=[lr_reducer, lr_scheduler, time_callback])

logg = LoggingMetrics(history_callback, time_callback)
logg.save_metrics_to_log(rootLogger)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
rootLogger.info('Test loss: %.4f' % scores[0])
rootLogger.info('Test accuracy: %.4f' % scores[1])
