#!/bin/bash
# exmple
# GPU, mxnet, resnet

epochs=5

system=gpu
docker_name=bench_gpu:0.1.0
model=mnist_mlp #resnet50
framework=tensorflow #mxnet
option_gpu=NVIDIA_VISIBLE_DEVICES=0

if [ "$framework" = "tensorflow" ]; then
    ver=1.8.0
    run_file=run_tf_backend.sh
    
elif [ "$framework" = "mxnet" ]; then
    ver=1.2.0
    run_file=run_mxnet_backend.sh       
fi

store=experiments/${system}_config/${framework}_${ver}/

## run_docker
docker run -it --name test --runtime=nvidia -e ${option_gpu} -e GRANT_SUDO=yes --user root -v ${PWD}/keras-apache-mxnet/benchmark/scripts:/home/work/ -d ${docker_name} /bin/bash

## run_benchmark
benchmark_sh="cd /home/work/ && mkdir -p ${store} && ./${run_file} ${system}_config ${model} False ${epochs}"
echo ${benchmark_sh}
docker exec test /bin/bash -c "${benchmark_sh}"

## move_result
mv_sh="cd /home/work/ && mv ${framework}_*.log ${store}"
echo ${mv_sh}
docker exec test /bin/bash -c "${mv_sh}"

## close docker
docker rm -f test
