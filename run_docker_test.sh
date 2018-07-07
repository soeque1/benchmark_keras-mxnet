#!/bin/bash
#rm -rf keras-apache-mxnet/benchmark/scripts/experiments/

iters=$1

for ((i=1;i<=$iters;i++)); 
do 

  echo $i
  ## GPU
  for system in cpu #4_gpu gpu cpu
    do
    for model in mnist_mlp # resnet50
      do
      for framework in mxnet # tensorflow
        do
          if [ "$system" = "cpu" ]; then
            docker_nm='mkl'
            n_gpu=0
            option_gpu=1
          elif [ "$system" = "4_gpu" ]; then
            docker_nm='gpu'
            n_gpu=4
            option_gpu=NVIDIA_VISIBLE_DEVICES=4,5,6,7
          elif [ "$system" = "gpu" ]; then
            docker_nm=$system
            n_gpu=1
            option_gpu=NVIDIA_VISIBLE_DEVICES=4
          fi

          docker run -it --name test --runtime=nvidia -e ${option_gpu} -e GRANT_SUDO=yes --user root -v ${PWD}/keras-apache-mxnet/benchmark/scripts:/home/work/ -d bench_${docker_nm}:0.1.0 /bin/bash

          if [ "$framework" = "tensorflow" ]; then
              ver=1.8.0
              run_file=run_tf_backend.sh
              channel_order=last
              
          elif [ "$framework" = "mxnet" ]; then
              ver=1.2.0
              run_file=run_mxnet_backend.sh
              channel_order=last              
          fi

          store=experiments/${system}_config/${framework}_${ver}/
          
          benchmark_sh="cd /home/work/ && mkdir -p ${store} && ./${run_file} ${system}_config $model False 1"
          echo ${benchmark_sh}
          docker exec test /bin/bash -c "${benchmark_sh}"
          
          mv_sh="cd /home/work/ && cat ${framework}_*.log >> ${store}/${i}_$(find ./ -name ${framework}_*.log -printf '%f\n') && rm -rf ${framework}_*.log"
          echo ${mv_sh}
          docker exec test /bin/bash -c "${mv_sh}"

          docker rm -f test
      done
    done
  done

done
