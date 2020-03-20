#!/bin/bash

# Stop the MPS control daemon for each GPU and clean up /home/aymen/MPS/
num_gpus=2

# Check if there is an alive MPS server on every GPU

for ((i=0; i< $num_gpus; i++)) # Iterate through the avilable devices

do


echo $i
export CUDA_MPS_PIPE_DIRECTORY=/home/aymen/MPS/tmp/mps_$i

echo "quit" | nvidia-cuda-mps-control

rm -fr /home/aymen/MPS/tmp/mps_$i
rm -fr /home/aymen/MPS/tmp/mps_log_$i

end do


done

