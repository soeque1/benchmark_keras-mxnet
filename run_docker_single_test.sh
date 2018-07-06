#!/bin/bash
# exmple
# GPU, mxnet, resnet

system=gpu
docker_name=bench_gpu:0.1.0
model=resnet50
framework=mxnet_1.2.0
option_gpu=NVIDIA_VISIBLE_DEVICE=0
store=experiments/${system}_config/${framework}/

iter=1

## docker run
docker run -it --name test --runtime=nvidia -e $option_gpu -e GRANT_SUDO=yes --user root -v ${PWD}/keras-apache-mxnet/benchmark/scripts:/home/work/ -d $docker_name /bin/bash

## run benchmark's scripts
docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $store && ./run_mxnet_backend.sh ${system}_config $model False $iter"

## close docker
docker rm -f test
