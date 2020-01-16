from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import csv
import os
from itertools import izip_longest
from operator import itemgetter

#benchmarks=['vgg16', 'alexnet', 'dcgan', 'squeezenet', 'resnet18']
stages=['req', 'batch', 'wait_exec','exec','cmp','send']
#batchs=[0.01]

def isEarlierTimestamp(org, dest):
    org_seconds=float(org.split(':')[-1]) / 1000 + float(org.split(':')[2]) + float(org.split(':')[1]) *60 + float(org.split(':')[0])*3600
    dest_seconds=float(dest.split(':')[-1]) / 1000 + float(dest.split(':')[2]) + float(dest.split(':')[1]) *60 + float(dest.split(':')[0])*3600
    if (org_seconds<=dest_seconds ):
        return True
    else: 
        return False
def diffTimestamp_s(start, end): # returns how much time has elapsed between two timestamp in ms
        start_seconds=float(start.split(':')[-1]) / 1000 + float(start.split(':')[2]) + float(start.split(':')[1]) *60 
        end_seconds=float(end.split(':')[-1]) / 1000 + float(end.split(':')[2]) + float(end.split(':')[1]) *60 
        if start_seconds > end_seconds:
                end_seconds = 3600 + end_seconds
        return (end_seconds - start_seconds)

def getThroughput_MakeSpan(vec): # accepts vector of timestamps
        if len(vec) == 0:
            return 0
        elif len(vec) == 1:
            return 0
        makespan = diffTimestamp_s(vec[0], vec[-1])
        if makespan==0:
            return 0
        return len(vec) / diffTimestamp_s(vec[0], vec[-1])
def getThroughput_MakeSpan2Phase(vec,timestamp): # accepts vector of timestamps & the timestamp to split the vector
    firstVec=[]
    secondVec=[]
    for i in range(len(vec)):
        if isEarlierTimestamp(vec[i], timestamp):
            firstVec.append(vec[i])
        else:
            secondVec.append(vec[i])
    return getThroughput_MakeSpan(firstVec), diffTimestamp_s(firstVec[0],timestamp)

def nonzeroAverage(vec):
	pvec = np.array([num for num in vec if num > 0])
	nvec=sorted(pvec)
        if len(nvec) ==0:
		return 0
	if len(nvec) ==1:
		return vec[0]
	return sum(nvec) / len(nvec)


def tail(vec):
	if len(vec) == 0:
		return 0
	elif len(vec) == 1:
		return vec[0]
        pvec = np.array([num for num in vec if num >= 0])
	nvec=sorted(pvec)
	return np.percentile(nvec, 99, interpolation='nearest')

def average(vec):
	if len(vec) ==0:
		return 0
	if len(vec) ==1:
		return vec[0]
	pvec = np.array([num for num in vec if num >= 0])
	nvec=sorted(pvec)
	del nvec[-1] # exclude the outliers
	return sum(nvec) / len(nvec)

def med(vec):
	pvec = vec[vec>=0]
	nvec=sorted(pvec)
	nlen=len(vec)
	if nlen ==0:
		return 0
	else:
		return np.percentile(nvec,50)
def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('log_file',
			help='file which log info')
	parser.add_argument('result_file',
			help='file which holds results')
	args = parser.parse_args()
	return args


def analyzeServerBreakdownExec(serv_breakdown_file, result_file, schedule_mode, benchmarks):
    server_data = []
    with open (serv_breakdown_file) as fp:
	print "processing " + serv_breakdown_file
	benchwise_info = {}
	for bench in benchmarks:
	    benchwise_info[bench] = {}
	for bench in benchmarks:
	    for stage in stages:
		benchwise_info[bench][stage]=[]
	    lines = fp.readlines()
	    cnt=1 # skip the first
	    for line in lines:
		 if cnt != 0:
			cnt = cnt - 1
			continue
		words=line.split(',')
			#if float(words[4]) > 10000 or float(words[5]) > 10000 : # we have a serious problem of huge numbers, skip it if so
			#	continue
		bench = words[1]
                benchwise_info[bench]['req'].append(float(words[4]))
                benchwise_info[bench]['batch'].append(float(words[5]))
                benchwise_info[bench]['wait_exec'].append(float(words[6]))
                benchwise_info[bench]['exec'].append(float(words[7]))
                benchwise_info[bench]['cmp'].append(float(words[8]))
                benchwise_info[bench]['send'].append(float(words[9]))
		for bench in benchmarks:
			item = (schedule_mode,bench)
			for stage in stages:
				item = item + (average(benchwise_info[bench][stage]),)
			server_data.append(item)
    with open(result_file,"a") as fp:
	sorted_data = sorted(server_data,key=itemgetter(0,1))
	fp.write("desc,exec(avg),exec(tail)\n");
	for item in sorted_data:
	    item_name=str(item[0])+"-"+str(item[1])
	    fp.write(item_name+",")
	    for x in range(2,len(item)):
		element = item[x]
		fp.write(str(element)+",")
	    fp.write("\n");


def readBenchmarks(server_log_file):
    benchmarks=[]
    with open (server_log_file) as fp:
        lines = fp.readlines()
        cnt=1
        for line in lines:
            if cnt != 0:
                cnt = cnt -1
                continue
            bench = line.split(',')[1]
            if bench not in benchmarks:
                benchmarks.append(bench)
    return benchmarks

def parsetoFile( log_file, result_file):
    client_benchwise_info={}
    benchmarks=readBenchmarks(log_file)
    for bench in benchmarks:
        client_benchwise_info[bench] = []
    name=f.split(".")[0]
    schedule=f.split("-")[0]
    if metric == "exec":
        analyzeServerExec(log_file, result_file,schedule, benchmarks)
def main():
    args = parse_args()
    #parameter_file="param.txt" # the file which holds model related parameters
    parsetoFile(args.log_file, args.result_file)
        

if __name__ == '__main__':
	main()

