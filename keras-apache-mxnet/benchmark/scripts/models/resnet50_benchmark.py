"""Benchmark a Resnet50 model
Credit:
Script modified from TensorFlow Benchmark repo:
https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/models/resnet50_benchmark.py
"""

from __future__ import print_function

import logging
import time

import numpy as np
from logging_metrics import LoggingMetrics
from models.timehistory import TimeHistory

import keras
from keras import backend as K


def crossentropy_from_logits(y_true, y_pred):
    return keras.backend.categorical_crossentropy(target=y_true,
                                                  output=y_pred,
                                                  from_logits=True)


class Resnet50Benchmark:

    def __init__(self):
        self.test_name = "resnet50"
        self.sample_type = "images"
        self.total_time = 0
        self.batch_size = 16
        self.epochs = 20
        self.num_samples = 100 #0
        self.test_type = 'tf.keras, eager_mode'

    def run_benchmark(self, gpus=0, inference=False, use_dataset_tensors=False, epochs=20):
        self.epochs = epochs
        if gpus > 1:
            self.batch_size = self.batch_size * gpus

        # prepare logging
        # file name: backend_data_format_dataset_model_batch_size_gpus.log
        log_file = K.backend() + '_' + K.image_data_format() + '_synthetic_resnet50_batch_size_' + str(self.batch_size) + '_' + str(gpus) + 'gpus.log'  # nopep8
        logging.basicConfig(level=logging.INFO, filename=log_file)

        print("Running model ", self.test_name)
        keras.backend.set_learning_phase(True)

        input_shape = (self.num_samples, 3, 256, 256)
        num_classes = 1000

        x_train = np.random.randint(0, 255, input_shape)
        y_train = np.random.randint(0, num_classes, (input_shape[0],))
        y_train = keras.utils.to_categorical(y_train, num_classes)

        if (keras.backend.backend() == "tensorflow" or keras.backend.backend() == "mxnet") and gpus >= 1:
            keras.backend.set_image_data_format('channels_first')

        if keras.backend.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            input_shape = (self.num_samples, 256, 256, 3)
        print("data format is ", keras.backend.image_data_format())
        print(x_train.shape)
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_train /= 255

        inputs = keras.layers.Input(shape=input_shape[1:])
        outputs = keras.applications.ResNet50(include_top=False,
                                              pooling='avg',
                                              weights=None, input_shape=input_shape[1:])(inputs)
        predictions = keras.layers.Dense(num_classes)(outputs)
        model = keras.models.Model(inputs, predictions)
        # use multi gpu model for more than 1 gpu
        if (keras.backend.backend() == "tensorflow" or keras.backend.backend() == "mxnet") and gpus > 1:
            model = keras.utils.multi_gpu_model(model, gpus=gpus, cpu_merge=False)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(lr=0.0001),
                      metrics=['accuracy'])
        time_callback = TimeHistory()
        callbacks = [time_callback]
        batch_size = self.batch_size * gpus if gpus > 0 else self.batch_size

        history_callback = model.fit(x_train, y_train, batch_size=batch_size, epochs=self.epochs,
                                     shuffle=True, callbacks=callbacks)

        logg = LoggingMetrics(history_callback, time_callback)
        logg.save_metrics_to_log(logging)
