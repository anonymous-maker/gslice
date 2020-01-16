#!/bin/bash

if [ -z $1 ]
then
        echo "no device was specified! please specify a device ex) startProxy.sh 0"
        exit
fi

if [ -z $2 ]
then
        echo "no threadcap was specified, using default 100%"
        cap=100

else
        echo "using threadcap "$2
        cap=$2
fi
if [ -z $3 ]
then
        echo "no dedup was specified, using default 0"
        dedup=0

else
        echo "using dedup "$3
        dedup=$3
fi



device=$1
COMMON_DIR=$PWD/../../pytorch-common/

SERVER_PID=$(echo get_server_list | nvidia-cuda-mps-control)

if [ -z $SERVER_PID ]
then
        echo "there is no mps server, make sure that you have turned on MPS Server, proxy not turned on"
        exit
fi


export CUDA_VISIBLE_DEVICES=$device
echo set_active_thread_percentage $SERVER_PID $cap | nvidia-cuda-mps-control
#echo get_active_thread_percentage $SERVER_PID | nvidia-cuda-mps-control

#if [ \( "$device" -eq 0 \) -o \( "$device" -eq 1 \) ]
if [ "$device" -eq 0 ]
then
    taskset -c 5-9 build/proxy --common $COMMON_DIR --devid $device  --threadcap $cap --dedup $dedup
else
    taskset -c 15-19 build/proxy --common $COMMON_DIR --devid $device  --threadcap $cap --dedup $dedup
fi

#revert to old state just in case
export CUDA_VISIBLE_DEVICES=0


