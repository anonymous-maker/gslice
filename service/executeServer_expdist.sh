#!/bin/bash


if [ -z $1 ]
then
echo "no scheduler was specified! default "no" will be used"
scheduler='no'
else
scheduler=$1
fi


if [ -z $2 ]
then
    echo "no number of GPU was specified! default 1 will be used"
    NGPUS=1
  else
    NGPUS=$2
fi

if [ -z $3 ]
then
    echo "no number of CPU was specified! default 1 will be used"
    NCPUS=1
  else
    NCPUS=$3
fi
if [ -z $4 ]
then
    echo "MPS option not turned on as fourth parameter is empty"
    MPS=0
  else
    MPS=1
fi

#ncores=$((20 / $NCPUS))
#ncores=6

#export CUDA_VISIBLE_DEVICES=0
#export OMP_NUM_THREADS=$ncores
#NGPUS=4

./build/djinn --common ../pytorch-common/ --weights weights/ --portno 8080 --gpu 1 --nets nets.txt --profile gpuprocinfo.csv \
        --ngpu $NGPUS -s $scheduler  --ncpu $NCPUS \
        --adaptive_batch 0 --local 1 --mps $MPS\
        --input_txt server_input.txt --req_gen_txt req_gen_spec.txt\
        --exp_dist 1
