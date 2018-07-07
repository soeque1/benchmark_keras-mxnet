#!/bin/bash
#rm -rf keras-apache-mxnet/benchmark/scripts/experiments/

iters=$1

for ((i=1;i<=$iters;i++)); 
do 

  echo $i
  ## GPU
  for system in gpu #cpu
    do
    for model in mnist_mlp # resnet50
      do
      for framework in mxnet # tensorflow
        do

          if [ "$system" = "cpu" ]; then
            docker_nm='mkl'
            n_gpu=0
          else
            docker_nm=$system
            n_gpu=1
          fi

          docker run -it --name test --runtime=nvidia -e GRANT_SUDO=yes --user root -v ${PWD}/keras-apache-mxnet/benchmark/scripts:/home/work/ -d -p 8888:8888 bench_$system:0.1.0 /bin/bash

          if [ "$framework" = "tensorflow" ]; then
              ver=1_8_0
              run_file=run_tf_backend.sh
              channel_order=last
              
          elif [ "$framework" = "mxnet" ]; then
              ver=1_2_0
              run_file=run_mxnet_backend.sh
              channel_order=last              
          fi

          res_dir=experiments/${system}_config/${framework}_${ver}/
          #echo $res_dir
          docker exec test /bin/bash -c "cd /home/work/ && mkdir -p $res_dir && ./$run_file ${system}_config ${model} True 1"
          docker exec test /bin/bash -c "cd /home/work/ && cat ${framework}_*.log >> ${res_dir}/${i}_$(find ./ -name ${framework}_*.log -printf '%f\n') && rm -rf ${framework}_*.log"
          docker rm -f test
      done
    done
  done

done
