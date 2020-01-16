#!/bin/bash


NGPUS=4
# Stop the MPS control daemon for each GPU and clean up /tmp

for ((i=0; i< $NGPUS; i++))
do
    echo "shutting down MPS server for GPU "$i
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
    sudo nvidia-smi -i $i -c DEFAULT
    echo "quit" | nvidia-cuda-mps-control
    rm -rf /tmp/mps_$i
    rm -rf /tmp/mps_log_$i
done

