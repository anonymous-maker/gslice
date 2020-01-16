#!/bin/bash

if [ -z $1 ]
then
echo "Please specify the number of cores for openblas"
exit 1
fi
##export OPENBLAS_NUM_THREADS=$1
#
##USE this one instead the one on the top!! 
export OMP_NUM_THREADS=$1

./djinn --common ../common/ --weights weights/ --portno 8080 --gpu 0 --debug 0 --nets nets.txt -s no
