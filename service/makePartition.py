#!/usr/bin/env python
import sys
from os import listdir
from numpy import median 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import csv
import os
import statistics
#from itertools import izip_longest
from operator import itemgetter

models=[] # vector that stores type of model
model_table={}
model_SLO={}
model_input_rate={}

#available batch sizes
batches=('1', '2', '4', '8','16','32')
caps=('20', '40', '60', '80', '100')


def readModelTable(model_file):
    # read models
    with open(model_file, "r") as fp:
        lines=fp.readlines()
        for line in lines:
            tokens=line.split(",")
            model=tokens[0]
            c0=float(tokens[1])
            c1=float(tokens[2])
            if model not in models:
                models.append(model)
            model_table[model]=[c0,c1]
    #for debugging
    #print(model_table)
def readSLOInfo(slo_file):
    # read models
    with open(slo_file, "r") as fp:
        lines=fp.readlines()
        for line in lines:
            tokens=line.split(",")
            model=tokens[0]
            SLO=float(tokens[1])
            model_SLO[model]=SLO
    #for debugging
    #print(model_SLO)
def readReqInfo(req_file):
    # read models
    with open(req_file, "r") as fp:
        lines=fp.readlines()
        for line in lines:
            tokens=line.split(",")
            model=tokens[0]
            input_rate=1/float(tokens[1]) #req/s
            ntasks=int(tokens[2])
            model_input_rate[model]=input_rate
    #for debugging
    #print(model_input_rate)

def parse_args():
    parser = ArgumentParser(description=__doc__,
            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_info',
            help='latency estimation model related info')
    parser.add_argument('nGPUs',
            help='number of GPU')
    parser.add_argument('request_spec',
            help='txt which holds specifcation of requests')
    parser.add_argument('slo',
            help='file which holds SLO related info')
    args = parser.parse_args()
    return args

def getLatency(model,batchsize,cap):
    return model_table[model][0]*(float(batchsize)/float(cap)) + model_table[model][1]*1.2

#returns fitting resource for given model and number of GPUs
def checkFittingResource(model,nGPUs):
    input_rate=model_input_rate[model]
    configs=[]
    #exhaustive search for best-fitting 
    for i in range(1,nGPUs+1):
        for batch in batches:
            for cap in caps:
                service_rate = (int(batch)/getLatency(model,int(batch),int(cap)))*1000*i
                if (input_rate <= service_rate):
                    #print("cap: "+cap+",batch: "+batch+",devnum: "+str(i))
                    configs.append((int(batch), int(cap),i))
    return configs


#chose best which fits SLO
def bestfitConfig(possible_configs,model):
    #simulate a run
    min_diff = model_SLO[model]
    isGuarantee=False
    for config in possible_configs:
        runtime=0
        #print(config)
        cap=config[1]
        batch=config[0]
        if model_SLO[model] > getLatency(model,batch,cap):
            isGuarantee = True
            if model_SLO[model] - getLatency(model,batch,cap) < min_diff:
                min_diff=model_SLO[model] - getLatency(model,batch,cap)
                theconfig=config
    
    if not isGuarantee:
        print("SLO can not be guaranteed for task "+model)
    return theconfig

def checkPerformance(partitions, models,nGPUs):
    satisfiable=True
    for model in models:
        for i in range(0,len(partitions)):
            for j in range(0,len(partitions[i])):
                demand=model_input_rate[model]- (32/getLatency(model,32,partitions[i][j]))*1000*nGPUs
                if demand > 0: #means we cannot satisfy SLO no matter what
                    satisfiable=False
                    print("task "+model+" is too demanding "+str(newdemand)+"tasks per sec will not be addressed")
                    exit(1)
    return satisfiable


def makePartition(nGPUs):
    partitions=[]
    nGPUs=int(nGPUs)
    remainingGPUs=nGPUs
    remainingmodels=models.copy()
    index=0
    #check whether tasks can be served when offered maximum resource
    while remainingGPUs != 0:
        demand=-100000000
        mostdemanding="none"
        for model in remainingmodels:
            newdemand=model_input_rate[model]- (32/getLatency(model,32,100))*1000*remainingGPUs
            print("model : "+model+"demand: "+str(newdemand))
            if newdemand > 0: #means we cannot satisfy SLO no matter what
                print("task "+model+" is too demanding for "+str(remainingGPUs)+"GPUs, "+str(newdemand)+"tasks per sec will not be addressed")
                exit(1)
            if newdemand > demand:
                demand=newdemand
                mostdemanding=model
        #first setup for most demanding resource 
        possibleconfig = checkFittingResource(mostdemanding,remainingGPUs)
        print(possibleconfig)
        #consider SLO among possible configuration and get best fitting partition
        config = bestfitConfig(possibleconfig,mostdemanding)
        print("best config as follows:")
        print(config)
        usedGPU=config[2]
        cap=config[1]
        for i in range(index, index+usedGPU):
            if cap != 100:
                partitions.append([cap,100-cap])
            else:
                partitions.append([100])
        index = index + usedGPU
        remainingGPUs = remainingGPUs - usedGPU
        remainingmodels.remove(mostdemanding)
        ret = checkPerformance(partitions, remainingmodels,nGPUs)
        if ret:
            print("good to go")
        else:
            print("You are in for one big headache...can not be satisfied with partitioned resource")
    return partitions

def writeAvailPartitions(partitions, output_file, nGPUs):
    with open(output_file, "w") as fp:
        for model in models:
            availCap=[]
            fp.write(model+",")
            for i in range(0,len(partitions)):
                for j in range(0,len(partitions[i])):
                    demand=model_input_rate[model]- (32/getLatency(model,32,partitions[i][j]))*1000*int(nGPUs)
                    latency = model_SLO[model] -getLatency(model,32,partitions[i][j])
                    if demand < 0 and latency > 0: #means we satisfy request rate
                        if partitions[i][j] not in availCap:
                            availCap.append(partitions[i][j])
            for cap in availCap:
                fp.write(str(cap)+",")
            fp.write("\n")
   
def writePartitions(partitions, output_file,nGPUs):
    pairs=[]
    with open(output_file, "w") as fp:
        for i in range(0,int(nGPUs)):
            for j in range(0,len(partitions[i])):
                pairs.append((i,partitions[i][j]))
        for pair in pairs:
            fp.write(str(pair[0])+","+str(pair[1])+"\n")


def main():
    args = parse_args()
    readModelTable(args.model_info)
    readReqInfo(args.request_spec)
    readSLOInfo(args.slo)
    results=makePartition(args.nGPUs)
    print("partition results: ")
    print(results)
    writeAvailPartitions(results,"AvailvGPU.txt",args.nGPUs)
    writePartitions(results,"ThreadCap.txt",args.nGPUs)
if __name__ == '__main__':
    main()
