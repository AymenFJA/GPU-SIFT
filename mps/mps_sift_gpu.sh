#!/bin/bash
# Script to run 2 local processes on the same GPU

export CUDA_VISIBLE_DEVICES=0
lrank=$OMPI_COMM_WORLD_LOCAL_RANK

case ${lrank} in

[0])
    export CUDA_MPS_PIPE_DIRECTORY=/home/aymen/MPS/mps_0
    /cudasift /pylon5/mc3bggp/aymen/geolocation_dataset_new/[ADAR]_orthoWV02_11FEB162145457-P1BS-1030010009B6F300_u08rf3031.tif 0 0 1000 1000 /pylon5/mc3bggp/aymen/geolocation_dataset_new/\[ADAR\]_orthoWV02_11FEB162145461-P1BS-1030010009B6F300_u08rf3031.tif 0 0 1000 1000
    ;;
[1])
    export CUDA_MPS_PIPE_DIRECTORY=/home/aymen/MPS/mps_1
   /cudasift /pylon5/mc3bggp/aymen/geolocation_dataset_new/[ADAR]_orthoWV02_11FEB162145457-P1BS-1030010009B6F300_u08rf3031.tif 0 0 1000 1000 /pylon5/mc3bggp/aymen/geolocation_dataset_new/\[ADAR\]_orthoWV02_11FEB162145461-P1BS-1030010009B6F300_u08rf3031.tif 0 0 1000 1000 
    ;;

esac
echo "done"
