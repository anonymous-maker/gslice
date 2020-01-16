#!/bin/bash

if [ -z $1 ]
then
    echo "please specify either org or extended"
    exit 1
fi

if [ $1 == 'org' ]
then
    cp MaxBatch_org.txt MaxBatch.txt
    cp speedup_org.txt speedup.txt
    cp nets_org.txt nets.txt
    cp weightedepoch_org.txt weightedepoch.txt
    cp priority_org.txt priority.txt
    echo "changed to orignal version"
elif [ $1 == 'extended' ]
then
    cp MaxBatch_extended.txt MaxBatch.txt
    cp speedup_extended.txt speedup.txt
    cp nets_extended.txt nets.txt
    cp weightedepoch_extended.txt weightedepoch.txt
    cp priority_extended.txt priority.txt
    echo "changed to extended version"
elif [ $1 == 'fully_extended' ]
then
    cp MaxBatch_fully_extended.txt MaxBatch.txt
    cp speedup_fully_extended.txt speedup.txt
    cp nets_fully_extended.txt nets.txt
    cp weightedepoch_fully_extended.txt weightedepoch.txt
    cp priority_fully_extended.txt priority.txt
    echo "changed to fully extended version"
else
    echo "wrong specification "
fi




