/*
 *This file is wrapper of a single request
 *
 * */

#ifndef REQUEST_H__
#define REQUEST_H__

#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "mon.h"
#include <torch/script.h>
#include "torchutils.h"
#include "state.h"

using namespace std;

class request{
private:
   int mTaskID; // the task ID that is managed by the server, for debugging purpose
    int reqID; // different from above, the ID that client has sent, made for profiling purpose 
  int mClientFD;
  int mBatchNum; // batch size
  int mDeviceID; // the device ID this request was allocated to, purely for debugging purpose
  djinn::TaskState mState;
  djinn::Backend mBackend;
  char mReqName[MAX_REQ_SIZE];

  uint64_t start;
  uint64_t endBatch;
  uint64_t startExec;
  uint64_t endExec;
  uint64_t endSend;
  uint64_t endReq;
  uint64_t endCmpq;

  
public:

  vector<torch::Tensor> _inputTensors; // in case model requires N tensor for inference, we maintain a vector of tensors

  request(int tid, int fd, int bnum)
 {
  mTaskID = tid;
  mClientFD= fd;
  mBatchNum = bnum;
 }

 ~request(){
  } 

void setState(djinn::TaskState s){
	mState = s;
}


djinn::TaskState getState(){
return mState;
}

void setReqName(char * net_name){
strcpy(mReqName, net_name);
}

char* getReqName(){
return mReqName; 
}

void* getInput(int i){
    if(mBackend == djinn::pytorch)
        return &_inputTensors[i];
}

int getTaskID(){
    return mTaskID;
}
int getClientFD(){
return mClientFD;
}

int getBatchNum(){
return mBatchNum;
}
/*
int getDataLen(){
return _input.size();
}*/

int setDeviceID(int id){
	mDeviceID = id;
}

void setReqID(int id){
	reqID=id;
}

void setBackend(djinn::Backend b){
    mBackend = b;
}
 
/*
void setInputData(float* inData, int Datalen){
    if (mBackend == djinn::caffe){
        _input.assign(inData,inData+Datalen);
    }
}*/
void pushInputData(torch::Tensor input){
        _inputTensors.push_back(input);
}

int getReqID(){
	return reqID;
}

void setStart(uint64_t time){
	start = time;
}
void setendBatch(uint64_t time){
	endBatch = time;
}
void setstartExec(uint64_t time){
	startExec = time;
}
void setendExec(uint64_t time){
	endExec = time;
}
void setendSend(uint64_t time){
	endSend = time;
}

void setendReq(uint64_t time){
    endReq = time;
}

void setendCmpq(uint64_t time){
    endCmpq = time;
}
uint64_t getStart(){ // used in FCFS scheduler
    return start;
}

#ifdef DEBUG                                                                                                                                                           
void printTimeStamps(){
    printf("Req ID : %d, start: %lu, endBatch: %lu, startExec: %lu, endExec: %lu, endSend: %lu \n", reqID, start, endBatch, startExec, endExec, endSend);
}
#endif
void writeToLog(FILE* fp){ 
    double reqTime = double(endReq - start);
    double execTime = double(endExec - startExec);
    double batchTime = double(endBatch-endReq);
    double prepTime = double(startExec - endBatch);
    double sendTime = double(endSend - endCmpq);
    double cmpTime = double(endCmpq - endExec);
    reqTime = reqTime / 1000000;
    execTime = execTime / 1000000;
    batchTime = batchTime / 1000000;
    prepTime = prepTime / 1000000;
    sendTime = sendTime / 1000000;
    cmpTime = cmpTime / 1000000;
    fprintf(fp,"%s,%s,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf\n",timeStamp(),mReqName,mTaskID,mDeviceID,reqTime,batchTime,prepTime,execTime,cmpTime,sendTime);
 //   fflush(fp);
    return;
}


};
#else
#endif 
