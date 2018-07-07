"""Benchmark a gluon model
Credit:
https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-gluon.html
"""

# import dependencies
from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)

import logging
import time
from logging_metrics import LoggingMetrics, LoggingMetricsCustom

class GluonCNNBenchmark:

    def __init__(self):
        self.test_name = "gloun_cnn"
        self.sample_type = "images"
        self.total_time = 0
        self.batch_size = 32
        self.epochs = 20
        self.num_samples = 10000
        self.test_type = 'tf.keras, eager_mode'
        
    def evaluate_accuracy(self, train_data, train_label, net, ctx):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(zip(train_data, train_label)):
            data = data.as_in_context(ctx)
            label = label.reshape(-1).as_in_context(ctx)
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]

    def run_benchmark(self, gpus=0, inference=False, use_dataset_tensors=False, epochs=20):
        self.epochs = epochs
        if gpus > 1:
            self.batch_size = self.batch_size * gpus
            
        if gpus == 0:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu()

        # prepare logging
        # file name: backend_data_format_dataset_model_batch_size_gpus.log
        log_file = 'mxnet' + '_synthetic_gluonn_cnn_batch_size_' + str(self.batch_size) + '_' + str(gpus) + 'gpus.log'  # nopep8
        logging.basicConfig(level=logging.INFO, filename=log_file)
        
        print("Running model ", self.test_name)
        
        num_inputs = 784
        num_outputs = 10
                                            
        images = mx.nd.random.uniform(0, 255, (self.num_samples, 1, 28, 28)).astype(dtype=np.float32)
        labels = mx.nd.random.uniform(0, 10, (self.num_samples)).astype(dtype=int)
        
        train_data = gluon.data.DataLoader(images, self.batch_size, shuffle=True)
        train_label = gluon.data.DataLoader(labels, self.batch_size, shuffle=True)
                                             
        num_fc = 512
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # The Flatten layer collapses all axis, except the first one, into one axis.
            net.add(gluon.nn.Flatten())
            net.add(gluon.nn.Dense(num_fc, activation="relu"))
            net.add(gluon.nn.Dense(num_outputs))
            
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
        
        print("Running model ", self.test_name)
        
        smoothing_constant = .01

        start = time.time()
        for e in range(self.epochs):
            for i, (data, label) in enumerate(zip(train_data, train_label)):
            
                data = data.as_in_context(ctx)
                label = label.reshape(-1).as_in_context(ctx)
                            
                with autograd.record():
                    output = net(data)
                    loss = softmax_cross_entropy(output, label)
                loss.backward()
                trainer.step(data.shape[0])

                ##########################
                #  Keep a moving average of the losses
                ##########################
                curr_loss = nd.mean(loss).asscalar()
                moving_loss = (curr_loss if ((i == 0) and (e == 0))
                               else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
       
        train_time = '%.2f ' % float(time.time() - start) + 'sec'
        train_times = [train_time]
                                     
        start = time.time()
        train_accuracy = self.evaluate_accuracy(train_data, train_label, net, ctx)
        infer_time = '%.2f ' % float(time.time() - start) + 'sec'
        infer_times = [infer_time]
        epochs = [self.epochs]

        #logg = LoggingMetrics(history_callback, time_callback)
        logg = LoggingMetricsCustom(train_times, infer_times, epochs)
        logg.save_metrics_to_log(logging)
