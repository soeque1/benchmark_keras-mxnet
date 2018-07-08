#!/bin/bash
# exmple
# GPU, mxnet, resnet

system=gpu
docker_name=bench_gpu:0.1.0
model=resnet50 #gluon_cnn #mnist_mlp #resnet50
framework=mxnet
ver=1.2.0
option_gpu=NVIDIA_VISIBLE_DEVICES=0
store=experiments/${system}_config/${framework}_${ver}/

epochs=20

## run_docker
docker run -it --name test --runtime=nvidia -e ${option_gpu} -e GRANT_SUDO=yes --user root -v ${PWD}/keras-apache-mxnet/benchmark/scripts:/home/work/ -d ${docker_name} /bin/bash

## run_benchmark
benchmark_sh="cd /home/work/ && mkdir -p ${store} && ./run_mxnet_backend.sh ${system}_config ${model} False ${epochs}"
echo ${benchmark_sh}
docker exec test /bin/bash -c "${benchmark_sh}"

## move_result
mv_sh="cd /home/work/ && mv ${framework}_*.log ${store}"
echo ${mv_sh}
docker exec test /bin/bash -c "${mv_sh}"

## close docker
docker rm -f test
