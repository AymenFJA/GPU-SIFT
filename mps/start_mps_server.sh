#!/bin/bash

num_gpus=2

# Start the MPS server for each GPU
for ((i=0; i< $num_gpus; i++)) # Iterate through the avilable devices 

do

mkdir /home/aymen/MPS/mps_$i  #MPS Deamon folder
mkdir /home/aymen/MPS/mps_log_$i #MPS Log folder

#For some reason MPS was not detecting any devices 
#So we had to set of the UUIDs
UUIDs=$(nvidia-smi -L)	
echo $UUIDs

export CUDA_VISIBLE_DEVICES= UUIDs #Make one of the GPUs visable per iteration

export CUDA_MPS_PIPE_DIRECTORY=/home/aymen/MPS/mps_$i # Set the path of the MPS Deamon folder 
export CUDA_MPS_LOG_DIRECTORY=/home/aymen/MPS/mps_log_$i # Set the path of the MPS logs folder 

nvidia-smi -i 2 -c EXCLUSIVE_PROCESS 


# Start the CUDA-MPS server
nvidia-cuda-mps-control -d&

done
