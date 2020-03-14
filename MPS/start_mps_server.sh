#!/bin/bash
#SBATCH -N 2
#SBATCH -p GPU
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:p100:2


num_gpus=2

# Start the MPS server for each GPU
for ((i=0; i< $num_gpus; i++)) # Iterate through the avilable devices 

do

mkdir /home/aymen/MPS/mps_$i  #MPS Deamon folder
mkdir /home/aymen/MPS/mps_log_$i #MPS Log folder

export CUDA_VISIBLE_DEVICES=$i #Make one of the GPUs visable per iteration

export CUDA_MPS_PIPE_DIRECTORY=/home/aymen/MPS/mps_$i # Set the path of the MPS Deamon folder 
export CUDA_MPS_LOG_DIRECTORY=/home/aymen/MPS/mps_log_$i # Set the path of the MPS logs folder 

nvidia-smi -i 2 -c EXCLUSIVE_PROCESS # It is not necessary. But it is a good idea to ensure all your CUDA processes access the GPU via MPS, 
                                     # rather than having some accidentally connecting to the GPU directly, or running multiple MPS servers by mistake.

# Start the CUDA-MPS server
srun --gres=gpu:2 nvidia-cuda-mps-control -d&

end do
