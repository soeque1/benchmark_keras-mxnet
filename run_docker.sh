export iter=$1

## GPU
docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v /home/hyungjunkim/Dropbox/repo/mxnet_blog/keras-apache-mxnet/benchmark/scripts:/home/work/ -d -p 8888:8888 bench_gpu:0.0.2 /bin/bash
res_dir=experiments/gpu_config/tensorflow_1_8_0/
docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir"
docker exec test /bin/bash -c "cd /home/work/ && ./run_tf_backend.sh gpu_config resnet50 True $iter && mv tensorflow_channels_last_synthetic_resnet50_batch_size_32_1gpus.log $res_dir"
docker rm -f test

docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v /home/hyungjunkim/Dropbox/repo/mxnet_blog/keras-apache-mxnet/benchmark/scripts:/home/work/ -d -p 8888:8888 bench_gpu:0.0.2 /bin/bash
res_dir=experiments/gpu_config/mxnet_1_2_0/
docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir"
docker exec test /bin/bash -c "cd /home/work/ && ./run_mxnet_backend.sh gpu_config resnet50 True $iter && mv mxnet_channels_first_synthetic_resnet50_batch_size_32_1gpus.log $res_dir"
docker rm -f test

docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v /home/hyungjunkim/Dropbox/repo/mxnet_blog/keras-apache-mxnet/benchmark/scripts:/home/work/ -d -p 8888:8888 bench_gpu_1:0.0.2 /bin/bash
res_dir=experiments/gpu_config/mxnet_1_1_0/
docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir"
docker exec test /bin/bash -c "cd /home/work/ && ./run_mxnet_backend.sh gpu_config resnet50 True $iter && mv mxnet_channels_first_synthetic_resnet50_batch_size_32_1gpus.log $res_dir"
docker rm -f test

## MKL
docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v /home/hyungjunkim/Dropbox/repo/mxnet_blog/keras-apache-mxnet/benchmark/scripts:/home/work -d -p 8888:8888 bench_mkl_1:0.0.2 /bin/bash
res_dir=experiments/cpu_config/mxnet-mkl_1_1_0/
docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir"
docker exec test /bin/bash -c "cd /home/work/ && ./run_mxnet_backend.sh cpu_config resnet50 True $iter && mv mxnet_channels_first_synthetic_resnet50_batch_size_32_0gpus.log $res_dir"
docker rm -f test

docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v /home/hyungjunkim/Dropbox/repo/mxnet_blog/keras-apache-mxnet/benchmark/scripts:/home/work -d -p 8888:8888 bench_mkl:0.0.2 /bin/bash
res_dir=experiments/cpu_config/mxnet-mkl_1_2_0/
docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir"
docker exec test /bin/bash -c "pip install mxnet==1.2.0b20180520 && pip install mxnet-mkl==1.2.0b20180520 && cd /home/work/ && ./run_mxnet_backend.sh cpu_config resnet50 True $iter && mv mxnet_channels_first_synthetic_resnet50_batch_size_32_0gpus.log $res_dir"
docker rm -f test

## CPU
docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v /home/hyungjunkim/Dropbox/repo/mxnet_blog/keras-apache-mxnet/benchmark/scripts:/home/work/ -d -p 8888:8888 bench_cpu:0.0.2 /bin/bash
res_dir=experiments/cpu_config/tensorflow_1_8_0/
docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir"
docker exec test /bin/bash -c "cd /home/work/ && ./run_tf_backend.sh cpu_config resnet50 True $iter && mv tensorflow_channels_last_synthetic_resnet50_batch_size_32_0gpus.log $res_dir"
docker rm -f test

docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v /home/hyungjunkim/Dropbox/repo/mxnet_blog/keras-apache-mxnet/benchmark/scripts:/home/work/ -d -p 8888:8888 bench_cpu:0.0.2 /bin/bash
res_dir=experiments/cpu_config/mxnet_1_2_0/
docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir"
docker exec test /bin/bash -c "cd /home/work/ && ./run_mxnet_backend.sh cpu_config resnet50 True $iter && mv mxnet_channels_first_synthetic_resnet50_batch_size_32_0gpus.log $res_dir"
docker rm -f test

docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v /home/hyungjunkim/Dropbox/repo/mxnet_blog/keras-apache-mxnet/benchmark/scripts:/home/work/ -d -p 8888:8888 bench_cpu:0.0.2 /bin/bash
res_dir=experiments/cpu_config/mxnet_1_1_0/
docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir"
docker exec test /bin/bash -c "cd /home/work/ && ./run_mxnet_backend.sh cpu_config resnet50 True $iter && mv mxnet_channels_first_synthetic_resnet50_batch_size_32_0gpus.log $res_dir"
docker rm -f test

echo $2 | sudo -S -k chown hyungjunkim:hyungjunkim -R ../keras-apache-mxnet/benchmark/scripts/experiments/
