#ifndef __BREQUEST
#define __BREQUEST
#include <queue>
#include <deque>
#include <vector>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "state.h"
#include "request.h"

using namespace std;

class batched_request{
private:
  int mBatchNum; // batch size
  int mGPUID;// gpuid to execute
  djinn::TaskState mstate;
  char _req_name[MAX_REQ_SIZE];
  
  int _max_batch; //maximum number of tasks that can be  batched 
  int mBatchID; // used for debugging


public:
vector<float> b_input;
#ifdef PYTORCH
vector<torch::Tensor> b_tensors; 
#endif 
vector<shared_ptr<request>> taskRequests;
batched_request() //makes empty batched_request
 {
  mBatchNum=0;
  mGPUID=-1;
 }

  ~batched_request(){} 

void setState(djinn::TaskState s){
	mstate = s;
}

void setBatchID(int id){
    mBatchID = id;
}

int getBatchID(){
    return mBatchID;
}

vector<shared_ptr<request>> getRequests(){
return taskRequests;
}

djinn::TaskState getState(){
return mstate;
}

char* getReqName(){
return _req_name; 
}

void setReqName(const char *net_name){
strcpy(_req_name, net_name);
}

int getBatchNum(){
return mBatchNum;
}


int getGPUID(){
return mGPUID;
}

void setGPUID(int gid){
mGPUID=gid;
}

void setMaxBatch(int size){
	_max_batch=size;
}

int getMaxBatch(){
	return _max_batch;
}

int getNTask(){
return taskRequests.size();
}

void addRequestInfo(shared_ptr<request> t, int typeofInputTensors){
if (taskRequests.size() > _max_batch){ 
 printf("number of batch exceeded MAX %d \n", _max_batch);
 exit(1);
}
taskRequests.push_back(t);
mBatchNum += t->getBatchNum();
#ifdef PYTORCH

    for (int i=0; i < typeofInputTensors; i++){
        vector<torch::Tensor> vecTensor;
        for(int j=0; j < t->getBatchNum(); j++) vecTensor.push_back(t->_inputTensors[i]);
        //if (b_tensors.size() < typeofInputTensors)
        b_tensors.push_back(torch::cat(vecTensor, 0));
        //else 
            //b_tensors[i] = torch::cat(vecTensor, 0);
    }
#endif
}


};

#endif 
