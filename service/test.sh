#!/bin/bash 

array=('slo' 'fifo')
value='slo'
if [[ " ${array[@]} " =~ " slo " ]]; then
            echo "its there"
fi

