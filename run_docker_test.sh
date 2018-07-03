export iter=$1

## GPU
for system in cpu gpu
  do
  for model in mnist_mlp resnet50
    do
    for framework in mxnet tensorflow
      do
        
        if [ "$system" = "cpu" ]; then
          docker_nm='mkl'
          n_gpu=0
        else 
          docker_nm=$system
          n_gpu=1
        fi
        
        docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v /home/hyungjunkim/Dropbox/repo/mxnet_blog/keras-apache-mxnet/benchmark/scripts:/home/work/ -d -p 8888:8888 bench_$system:0.1.0 /bin/bash
        
        if [ "$framework" = "tensorflow" ]; then
            ver=1_8_0
            run_file=run_tf_backend.sh
            channels_order=last
        elif [ "$framework" = "mxnet" ]; then
            ver=1_2_0
            run_file=run_mxnet_backend.sh
            channels_order=first
        fi
        
        res_dir=experiments/'$system'_config/'$framework'_'$ver'/
        docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir && ./$run_file '$system'_config $model True $iter && mv '$framework'_channels_'$channels_order'_synthetic_'$model'_batch_size_32_'$n_gpu'gpus.log $res_dir"
        docker rm -f test
    done
  done
done
