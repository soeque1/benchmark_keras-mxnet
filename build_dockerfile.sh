#!/bin/sh
img_ver=0.1.0

docker build -t bench_gpu:$img_ver -f docker/bench_gpu/Dockerfile ../
docker build -t bench_cpu:$img_ver -f docker/bench_cpu/Dockerfile ../
docker build -t bench_mkl:$img_ver -f docker/bench_mkl/Dockerfile ../
