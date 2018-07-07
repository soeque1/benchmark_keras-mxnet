#!/bin/bash
# exmple
# GPU, mxnet, resnet

system=gpu
docker_name=bench_gpu:0.1.0
model=mnist_mlp #gluon_cnn #mnist_mlp #resnet50
framework=mxnet
ver=1.2.0
option_gpu=NVIDIA_VISIBLE_DEVICE=0
store=experiments/${system}_config/${framework}_${ver}/

iter=1

## run_docker
docker run -it --name test --runtime=nvidia -e ${option_gpu} -e GRANT_SUDO=yes --user root -v ${PWD}/keras-apache-mxnet/benchmark/scripts:/home/work/ -d ${docker_name} /bin/bash

## run_benchmark
benchmark_sh="cd /home/work/ && mkdir -p ${store} && ./run_mxnet_backend.sh ${system}_config $model False $iter"
echo ${benchmark_sh}
docker exec test /bin/bash -c "${benchmark_sh}"

## mv_result
mv_sh="cd /home/work/ && mv ${framework}_*.log ${store}"
echo ${mv_sh}
docker exec test /bin/bash -c "${mv_sh}"

## close docker
docker rm -f test
