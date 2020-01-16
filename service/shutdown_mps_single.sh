#!/bin/bash

for i in `seq 0 3`
do
sudo nvidia-smi -i $i -c DEFAULT
done
echo quit | nvidia-cuda-mps-control

