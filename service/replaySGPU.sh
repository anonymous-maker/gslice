#!/bin/bash
finish_flag=0
accept_cnt=0
total_iter=5
failed_cnt=0
if [ -z "$1" ]
then
    echo "please enter scen number"
      exit
else
    scen=$1
fi 

#copy appropriate req_gen_file
cp req_files/scen"$scen".txt req_gen_spec.txt 
while [ $accept_cnt -ne $total_iter ]
do
#reset mps
./shutdown_mps_single.sh 
./start_mps_single_server_multiGPU.sh
if [ $scen -eq 1 ] 
then
echo "uniform!"
./executeServer.sh wrr_sgpu 2 1 &
else
./executeServer_expdist.sh wrr_sgpu 2 1 &

fi
pid=$!
sleep 60
if ps -p $pid > /dev/null 
then
    failed_cnt=$((failed_cnt + 1))
    echo "failed attemp :  "$failed_cnt
	pkill djinn
else
echo "success!"
 accept_cnt=$((accept_cnt + 1))
failed_cnt=0
./copyResultsTo.sh wrr_sgpu $accept_cnt $scen
fi
done
