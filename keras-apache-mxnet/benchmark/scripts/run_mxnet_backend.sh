#!/bin/bash

# Keras MXNet Backend
# Credit:
# Script modified from TensoFlow Benchmark repo:
# https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/run_tf_backend.sh

######################################
#
# USAGE:
#       $ bash run_mxnet_backend.sh $mode $model $inference_mode $epochs
#
# SUPPORTED VALUES:
#       mode: cpu_config, gpu_config, 4_gpu_config, 8_gpu_config
#       models: resnet50, lstm_synthetic, lstm_nietzsche, lstm_wikitext2
#       inference_mode: True, False
#       epochs: Int. Ex: 20
# EXAMPLE:
#       $ bash run_mxnet_backend.sh cpu_config lstm_synthetic False 5
#
#####################################

python -c "from keras import backend"
KERAS_BACKEND=mxnet
sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"

# Use "cpu_config", "gpu_config", "4_gpu_config", and "8_gpu_config" as command line arguments to load the right
# config file.

# Supported models='resnet50 resnet50_tf_keras lstm_synthetic lstm_nietzsche lstm_wikitext2'

dir=`pwd`

python $dir/run_benchmark.py  --pwd=$dir --mode="$1" --model_name="$2" --dry_run=True --inference="$3" --epochs="$4"