#!/bin/bash
# number of GPUs
NGPUS=4

for((i=0; i< $NGPUS; i++))
do
mkdir -p /tmp/mps_$i
mkdir -p /tmp/mps_log_$i
export CUDA_VISIBLE_DEVICES=$i
sudo nvidia-smi -i $i -c EXCLUSIVE_PROCESS

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i # Select a location that’s accessible to the given $UID

export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$i # Select a location that’s accessible to the given $UID

nvidia-cuda-mps-control -d # Start the daemon
done
