#!/bin/bash

cp log.txt ../client/test_dir/test/server-exec.csv
cp queue_log.txt ../client/test_dir/test/server-queue.csv
cd ../client
rm test.csv 
python analyzeScheduleResults.py test_dir test.csv 
scp test.csv local:~/
