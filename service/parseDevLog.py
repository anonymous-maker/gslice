#!/usr/bin/env python
import sys
from os import listdir
from numpy import median 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import numpy as np
import pandas as pd
import csv
import os
#from itertools import izip_longest
from operator import itemgetter

possible_benchmarks=['alexnet', 'resnet18', 'vgg16', 'squeezenet', 'dcgan-gpu']

def isEarlierTimestamp(org, dest):
    org_seconds=float(org.split(':')[-1]) / 1000 + float(org.split(':')[2]) + float(org.split(':')[1]) *60 + float(org.split(':')[0])*3600
    dest_seconds=float(dest.split(':')[-1]) / 1000 + float(dest.split(':')[2]) + float(dest.split(':')[1]) *60 + float(dest.split(':')[0])*3600
    if (org_seconds<=dest_seconds ):
        return True
    else: 
        return False


def addTimeStamp(time,val): # add 'val' ms to timestamp
    start_ms = float(time.split(':')[-1]) *1000 + int(time.split(':')[1])*60*1000
    new_ms = start_ms + val 
    minutes=int(new_ms /(60*1000))
    remain=new_ms%(60*1000)
    seconds=int(remain/1000)
    remain=remain % 1000
    ms=int(remain)
    new_time_stamp = time.split(':')[0]
    new_time_stamp=new_time_stamp+":"+str(minutes)+":"+str(seconds)+":"+str(ms)
    return new_time_stamp

def diffTimestamp_s(start, end): # returns how much time has elapsed between two timestamp in seconds
        start_seconds=float(start.split(':')[-1]) + float(start.split(':')[1]) *60 
        end_seconds=float(end.split(':')[-1]) + float(end.split(':')[1]) *60 
        if start_seconds > end_seconds:
                end_seconds = 3600 + end_seconds
        return (end_seconds - start_seconds)

def average(vec):
    if len(vec) == 0:
        return 0
    #print(float(sum(vec)))
    return float(sum(vec)) / len(vec)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def isNaN(num):
    return num != num

def fillBenchmarks(log_file,benchmarks):
    with open(log_file, "r") as fp:
        lines = fp.readlines()
        cnt=200 #for reading the first 200 lines
        for line in lines:
            if cnt == 0:
                break
            cnt = cnt -1

def searchKeyandReturn(line, key):
    words=line.split(' ')
    #print words 
    for word in words:
        if len(word) !=0 and word[-1] == '\n':
            word = word[:-1]# exclude '\n'
        if word in possible_benchmarks:
            bench=word
            break
    #print bench
    if key == 'total':
        words[-1]=words[-1][:-1]         
        for word in words:
            if word.isdigit(): 
                value=word
                break
        return bench , int(value)
    elif key == 'Arrival':
        if len(words) != 8:
            value=-1
            return bench, value
        for word in words:
            if isfloat(word):
                value=word
                break
        value=float(value)
        if value > 1000 or value < 0 or isNaN(value): # very unlikey to happen
            value=-1
        return bench, value

def parseLog(log_file):
    data = {} # this will be a set which holds data vector for each bencmark
    timestamps=[]

    benchmarks = possible_benchmarks
    set_start_time=False
    #print "started parsing "+log_file
    for bench in benchmarks:
        data[bench]=[]
    with open(log_file,"r") as fp:
            lines = fp.readlines()
            for line in lines:
                line=line[:-1] # exclude '\n'
                words=line.split(" ")
                #print words
                if not set_start_time:
                    start_time=words[3]
                    set_start_time=True
                end_time=words[3]
                if words[-2] == "START":
                    s_time = words[3]
                    benchmark=words[0][1:-1]
                    continue
                if words[-2] == "END":
                    e_time = words[3]
                    diff_float_second  = diffTimestamp_s(s_time, e_time)
                    diff_int_ms = int(diff_float_second * 1000)
                    for i in range(0,diff_int_ms):
                        ret_time = addTimeStamp(s_time,i)
                        #print ret_time
                        timestamps.append(ret_time)
                        for b in benchmarks:
                            if b != benchmark:
                                data[b].append(0)
                            else:
                                data[benchmark].append(diff_int_ms)
    filen= log_file.split("/")[-1]
    filename=filen.split(".")[0] 
    output_file=filename+"_time_series.csv"
    #print "writing results to "+output_file
    with open(output_file,"w") as fp:
        fp.write("time,");
        for bench in benchmarks:
            fp.write(bench+",")
        fp.write("\n")
        for i in range(0,len(timestamps)):
            fp.write(timestamps[i]+",")
            for bench in benchmarks:
                if (data[bench][i] == 0):
                    fp.write(",")
                else:
                    fp.write(str(data[bench][i])+",")
            fp.write("\n");	
# leaving the argument function and main function for debugging
def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('log_dir',
			help='directory which hold the logs')
        args = parser.parse_args()
	return args
def main():
	args = parse_args()
        for f in os.listdir(args.log_dir):
            if "dev_log" not in f:
                continue
            parseLog(args.log_dir+"/"+f)
if __name__ == '__main__':
	main()
