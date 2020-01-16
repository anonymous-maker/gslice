#!/bin/bash

SERVER_DIR=$PWD
PROXY_DIR=$SERVER_DIR/proxy
PROXY_SH=startProxy.sh 

threadcap_file='ThreadCap.txt'
export CUDA_VISIBLE_DEVICES=0,1 # list all the devices that can be used for MPS

skip=1 # to skip the first line
IFS=","

#cat $threadcap_file
rm /tmp/gpusock*
prev_dev=0
prev_cap=0
while read dev cap
do
    echo $cap" on device "$dev
    cd $PROXY_DIR
    if [ "$prev_dev" -eq "$dev" -a "$prev_cap" -eq "$cap" ]
    then 
        echo "same"
        $PROXY_DIR/$PROXY_SH $dev $cap 1 & 
    else
       $PROXY_DIR/$PROXY_SH $dev $cap 0 & 
        echo "different"
    fi
    prev_dev=$dev
    prev_cap=$cap
    sleep 12
done < $threadcap_file
