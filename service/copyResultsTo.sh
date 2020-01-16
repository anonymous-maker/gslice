#!/bin/bash


if [ -z $1 ]
then
    echo "enter scheduler name"
    exit 
else
    sched=$1
fi

if [ -z $2 ]
then
    echo "enter iter number"
    exit 
else
    iter=$2

fi
if [ -z $3 ]
then
    echo "enter schenario number, ex) scen4 -> just type 4"
    exit
else
    scen="scen"$3
fi


# might need to change ROOT_RESULT_DIR
ROOT_RESULT_DIR=../client/raw_results3
RESULT_DIR=$ROOT_RESULT_DIR/$scen/$iter/$sched
mkdir -p $RESULT_DIR
 
cp log.txt $RESULT_DIR/server-exec.csv
cp queue_log.txt $RESULT_DIR/server-queue.csv

echo "Files copied to "$RESULT_DIR 
#cd ../client
#rm test.csv 
#python analyzeScheduleResults.py test_dir test.csv 
#scp test.csv local:~/
