#!/bin/bash

# This script is doing the following :
# Start the MPS server 
# Call the GPU-SIFT script
# Stop the MPS server

echo "Starting"

/bin/bash start_mps_server.sh 

/bin/bash mps_sift_gpu.sh

/bin/bash stop_mps_server.sh

echo "Stoping"
echo "Done"
