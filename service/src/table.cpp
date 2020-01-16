#include<iostream>
#include<algorithm>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdlib>
#include<vector>
#include<unordered_map>
#include<tuple>

#include "table.h"
using namespace std;

PerfTable::PerfTable(){
}

PerfTable::~PerfTable(){

}
double linearInterpolation(double x1, double y1, double x2, double y2, double p){
    return (((y2-y1)/(x2-x1))*(p-x1))+y1;
}

bool PerfTable::checkWorkload(string s){
    for(vector<string>::const_iterator i=benchType.begin(); i!=benchType.end(); ++i){
        if (s.compare(*i)==0){
            return true;
        }
    }
    return false;
}

void setupbenchType(string filename){
    

}

pair<int, int> returnTwoPoints(int i, vector<int> v){
    int s=v.size();
    if(i<=v.front()){
        return make_pair(v[0], v[1]); 
    }
    else if (i>=v.back()){
        return make_pair(v[s-2], v[s-1]);
    }
    for(int j=0; j<s-1; j++){
        if(i>v[j]&&i<v[j+1]){
            return make_pair(v[j], v[j+1]);
        }
    }
}

string concatAsKeyCPU(string bench, int batch, int thread){
    stringstream ss, ss2;
    ss<<batch;
    ss2<<thread;
    return bench+ss.str()+ss2.str();
}

string concatAsKeyGPU(string bench, int batch, int interference){
    stringstream ss, ss2;
    ss<<batch;
    ss2<<interference;
    return bench+ss.str()+ss2.str();
}

double PerfTable::findLatencyCPU(string bench, int batch, int thread){
    bool batchExists=false;
    bool threadExists=false;
    pair<int, int> batchPair;
    pair<int, int> threadPair;
    double linintFirst;
    double linintSecond;
    if (batch<1||thread<1||!checkWorkload(bench)){
        return -1;
    }
    if(find(threadNum.begin(), threadNum.end(), thread)!=threadNum.end()){
        threadExists=true;
    }
    if(find(batchSizeCPU.begin(), batchSizeCPU.end(), batch)!=batchSizeCPU.end()){
        batchExists=true;
    }
    if (batchExists&&threadExists){
        return CPUTable[concatAsKeyCPU(bench, batch, thread)];
    }
    else if(batchExists&&!threadExists){
        threadPair=returnTwoPoints(thread, threadNum);
        return linearInterpolation((double)threadPair.first, CPUTable[concatAsKeyCPU(bench, batch, threadPair.first)], (double)threadPair.second, CPUTable[concatAsKeyCPU(bench, batch, threadPair.second)], (double)thread);
    }
    else if(!batchExists&&threadExists){
        batchPair=returnTwoPoints(batch, batchSizeCPU);
        return linearInterpolation((double)batchPair.first, CPUTable[concatAsKeyCPU(bench, batchPair.first, thread)], (double)batchPair.second, CPUTable[concatAsKeyCPU(bench, batchPair.second, thread)], (double)batch);
    }
    else{
        threadPair=returnTwoPoints(thread, threadNum);
        batchPair=returnTwoPoints(batch, batchSizeCPU);
        linintFirst=linearInterpolation((double)batchPair.first, CPUTable[concatAsKeyCPU(bench, batchPair.first, threadPair.first)], (double)batchPair.second, CPUTable[concatAsKeyCPU(bench, batchPair.second, threadPair.first)], (double)batch);
        linintSecond=linearInterpolation((double)batchPair.first, CPUTable[concatAsKeyCPU(bench, batchPair.first, threadPair.second)], (double)batchPair.second, CPUTable[concatAsKeyCPU(bench, batchPair.second, threadPair.second)], (double)batch);
        return linearInterpolation((double)threadPair.first, linintFirst, (double)threadPair.second, linintSecond, thread);
    }
}

double PerfTable::findValueGPU(string bench, int batch, int interference){
    bool batchExists=false;
    bool interferenceExists=false;
    pair<int, int> batchPair;
    pair<int, int> interferencePair;
    double linintFirst;
    double linintSecond;
    if (batch<1||interference<0||!checkWorkload(bench)){
        return -1;
    }
    if(find(interferenceAmount.begin(), interferenceAmount.end(), interference)!=interferenceAmount.end()){
        interferenceExists=true;
    }
    if(find(batchSizeGPU.begin(), batchSizeGPU.end(), batch)!=batchSizeGPU.end()){
        batchExists=true;
    }
    if (batchExists&&interferenceExists){
        return GPUTable[concatAsKeyGPU(bench, batch, interference)];
    }
    else if(batchExists&&!interferenceExists){
        interferencePair=returnTwoPoints(interference, interferenceAmount);
        return linearInterpolation((double)interferencePair.first, GPUTable[concatAsKeyGPU(bench, batch, interferencePair.first)], (double)interferencePair.second, GPUTable[concatAsKeyGPU(bench, batch, interferencePair.second)], (double)interference);
    }
    else if(!batchExists&&interferenceExists){
        batchPair=returnTwoPoints(batch, batchSizeGPU);
        return linearInterpolation((double)batchPair.first, GPUTable[concatAsKeyGPU(bench, batchPair.first, interference)], (double)batchPair.second, GPUTable[concatAsKeyGPU(bench, batchPair.second, interference)], (double)batch);
    }
    else{
        interferencePair=returnTwoPoints(interference, interferenceAmount);
        batchPair=returnTwoPoints(batch, batchSizeGPU);
        linintFirst=linearInterpolation((double)batchPair.first, GPUTable[concatAsKeyGPU(bench, batchPair.first, interferencePair.first)], (double)batchPair.second, GPUTable[concatAsKeyGPU(bench, batchPair.second, interferencePair.first)], (double)batch);
        linintSecond=linearInterpolation((double)batchPair.first, GPUTable[concatAsKeyGPU(bench, batchPair.first, interferencePair.second)], (double)batchPair.second, GPUTable[concatAsKeyGPU(bench, batchPair.second, interferencePair.second)], (double)batch);
        return linearInterpolation((double)interferencePair.first, linintFirst, (double)interferencePair.second, linintSecond, interference);
    }
}

void PerfTable::createTableCPU(string filename){
    string line;
    string c;
    vector<string>* vec;
    string bench;
    tuple<string, int, int> tup;
    int batch;
    int thread;
    double latency;
    ifstream f(filename);
    if(f.is_open()){
        while(getline(f,line)){
            istringstream s(line);
            vec=new vector<string>;
            while(getline(s, c, ',')){
                vec->push_back(c);
            }
            bench=vec->at(0);
            batch=atoi(vec->at(1).c_str());
            
            if(find(benchType.begin(), benchType.end(),bench ) == benchType.end()){
                benchType.push_back(bench);
            }
            if(find(batchSizeCPU.begin(), batchSizeCPU.end(), batch)==batchSizeCPU.end()){
                batchSizeCPU.push_back(batch);
            }
            thread=atoi(vec->at(2).c_str());
            if(find(threadNum.begin(), threadNum.end(), thread)==threadNum.end()){
                threadNum.push_back(thread);
            }
            latency=atof(vec->at(3).c_str());
            delete vec;
            CPUTable[concatAsKeyCPU(bench, batch, thread)]= latency;
        }
        f.close();
    }
    sort(batchSizeCPU.begin(), batchSizeCPU.end());
    sort(threadNum.begin(), threadNum.end());
}

void PerfTable::createTableGPU(string filename){
    string line;
    string c;
    vector<string>* vec;
    string bench;
    int batch;
    int interference;
    double latency;
    ifstream f(filename);
    if(f.is_open()){
        while(getline(f,line)){
            istringstream s(line);
            vec=new vector<string>;
            while(getline(s, c, ',')){
                vec->push_back(c);
            }
            bench=vec->at(0);
            batch=atoi(vec->at(1).c_str());
            if(find(benchType.begin(), benchType.end(),bench ) == benchType.end()){
                benchType.push_back(bench);
            }

            if(find(batchSizeGPU.begin(), batchSizeGPU.end(), batch)==batchSizeGPU.end()){
                batchSizeGPU.push_back(batch);
            }
            interference=atoi(vec->at(2).c_str());
            if(find(interferenceAmount.begin(), interferenceAmount.end(), interference)==interferenceAmount.end()){
                interferenceAmount.push_back(interference);
            }
            latency=atof(vec->at(3).c_str());
            delete vec;
            GPUTable[concatAsKeyGPU(bench, batch, interference)]=latency;
        }
        f.close();
    }
    sort(batchSizeGPU.begin(), batchSizeGPU.end());
    sort(interferenceAmount.begin(), interferenceAmount.end());

}

void PerfTable::printTableContents(){
    for (int i =0; i < benchType.size(); i++){
        string bench = benchType[i];
        printf("printing info of %s for GPU\n",bench.c_str() );
        printf("batch size : 1 , interference: 0, latency : %lf \n", findValueGPU(bench,1,0));
        printf("batch size : 32 , interference: 0 , latency : %lf \n",findValueGPU(bench,32,0));
    }
}
