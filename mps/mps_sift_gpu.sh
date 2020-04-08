#!/bin/bash
# Script to run 2 local processes on the same GPU

export CUDA_VISIBLE_DEVICES=0
lrank=$OMPI_COMM_WORLD_LOCAL_RANK

case ${lrank} in

[0])
    export CUDA_MPS_PIPE_DIRECTORY=/home/aymen/MPS/mps_0
    ../cudasift /home/aymen/imagery/datasets/20160903222711_10300.tif 0 0 5000 5000 /aymen/imagery/datasets/20160903222711_10311.tif 0 0 5000 5000
    ;;
[1])
    export CUDA_MPS_PIPE_DIRECTORY=/home/aymen/MPS/mps_1
    ../cudasift /home/aymen/imagery/datasets/20160903222711_10320.tif 0 0 5000 5000 /aymen/imagery/datasets/20160903222711_10321.tif 0 0 5000 5000
    ;;

esac
echo "done"
