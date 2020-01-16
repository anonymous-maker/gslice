#!/bin/bash


if [ -z $1 ]
then
echo "Please specify the number of GPUs to monitor"
exit
else
GPU_DEVICES=$1

fi

REMOTE_URL=deep5.kaist.ac.kr

# DIRS
ROOT_DIR=$HOME/org/djinn_csb
SERVER_DIR=$ROOT_DIR/service
CLIENT_DIR=$ROOT_DIR/client


#SERVER SCRIPTS


MON_DIR=$HOME/git/monitor/cpp3	#directory which holds monitoring binary 
MON_BIN=nvml_mon				#name of monitoring binary

SET=$(seq 0 $(($GPU_DEVICES - 1)))

for id in $SET
do
$MON_DIR/$MON_BIN $SERVER_DIR $id gpu-$id.txt > /dev/null &
echo $!
done
