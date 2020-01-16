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

possible_benchmarks=['alexnet', 'resnet18', 'vgg16', 'squeezenet', 'dcgan-gpu', 'dcgan-cpu', 'dcgan']

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

def parseLog(log_file,output_file):
    data = {} # this will be a set which holds data vector for each bencmark
    #benchmarks=[]
    #fillBenchmarks(log_file,benchmarks)
    #benchmarks = possible_benchmarks
    #print benchmarks
    #for bench in benchmarks:
    #    data[bench]=[]
    KEY="SEND_ACK"
    time_dict = {}
    time_dict_keys=[]
    with open(log_file,"r") as fp:
            lines = fp.readlines()
            for line in lines:
                #print line
                if KEY in line:
                    words=line.split(' ')
                    key=words[1]
                    if key not in time_dict_keys:
                        time_dict_keys.append(key)
                        time_dict[key]=[]
                    time_dict[key].append(float(words[-2]))
    data=[]
    for key in time_dict_keys:
        item=(key, sum(time_dict[key]),average(time_dict[key]), max(time_dict[key]))
        data.append(item)
    sorted_data = data
    with open(output_file,"w") as fp:
        fp.write("range,sum(us),average(us),max(us) \n");
        for item in sorted_data:
            for x in range(0,len(item)):
                element = item[x]
                fp.write(str(element)+",")
            fp.write("\n");	
# leaving the argument function and main function for debugging
def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('log_file',
			help='debugging log file')
	parser.add_argument('output_file',
			help='file to store output vector')
	args = parser.parse_args()
	return args
def main():
	args = parse_args()
        parseLog(args.log_file,args.output_file)
if __name__ == '__main__':
	main()
