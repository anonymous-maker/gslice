#!/bin/bash

batches=('b1' 'b2' 'b4' 'b8' 'b16' 'b32')

# 0. make backup
cp MaxBatch.txt MaxBatch.txt_bak

# 1. no MPS proxy 
#for batch in "${batches[@]}"
#do
    
#    cp "$batch"_MaxBatch.txt MaxBatch.txt
#    cat MaxBatch.txt
#    ./executeServer.sh no 1 1
#    ./miniAnalyze.sh
#done

# 2. MPS proxy
./start_mps_single_server_multiGPU.sh
for batch in "${batches[@]}"
do
    ./setupProxys.sh
    cp "$batch"_MaxBatch.txt MaxBatch.txt
    cat MaxBatch.txt
    ./executeServer.sh no 1 1 1 
    ./miniAnalyze.sh
    pkill proxy  
done
./shutdown_mps_single.sh

#3. recover backup
mv MaxBatch.txt_bak MaxBatch.txt
