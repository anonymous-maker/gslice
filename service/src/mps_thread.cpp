#include <torch/script.h> // One-stop header.
#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <memory>
#include <sys/time.h>
#include <pthread.h>
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <queue>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "socket.h"
#include "tonic.h"
#include "cvmat_serialization.h"
#include "img_utils.h"
#include "torchutils.h"
#include "cpuUsage.h"
#include "common_utils.h" //printTimeStamp moved to her
#include "gpu_proxy.h"

#include "timer.h"
#include "thread.h"
#include "ts_list.h"
#include "state.h"
#include "request.h"
#include "batched_request.h"
#include "scheduler.h"
#include "mon.h"


#ifdef DEV_LOG
//2020-1-8 : added to print proxy execution LOG
map<pair<int,int>,FILE*> devlogTable;

extern string str_START;
extern string str_END;
#endif 

//2019-1-3 : fixed cores for each device 
#ifdef FIXED_CORES
int gpucores[4][2]={{0,1},{0,1},{10,11},{10,11}};
int cpucores[16]={2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19};
#endif 

// sets how many tasks can be scheduled/batching on the device at the same time
#define MAX_CPU_TYPE 2
#define MAX_GPU_TYPE 2
// sets how many tasks can run on the device at the same time 
#define MAX_GPU_RUN_SLOT_MPS 2
#define MAX_CPU_RUN_SLOT 1

// variables realted to main.cpp 
extern GlobalScheduler gsc;
extern string NET_LIST;
extern string COMMON;
extern string WEIGHT_DIR;
extern bool needScheduling;
extern bool USE_MPS;

// exter variables realated to server_thread.cpp 
extern SysInfo ServerState;

//synch related extern variables
extern mutex schedMtx;
extern mutex initMtx;

extern mutex completeMtx;
extern condition_variable completeCV;
extern map<std::string, mutex*> cmpTableMtx;;

//condition variables and vector needed for batching threads
map <pair<int,int>,condition_variable*> perProxyBatchCV;
map<pair<int,int>,mutex*> perProxyBatchMtx;
map<pair<int,int>,mutex*> perProxyExecMtx;

// following are synch variables for 
extern map<string, mutex*> perTaskBatchingMtx;
extern map<string, bool> perTaskisWaiting;


//used for signaling idle device threads 
extern vector<condition_variable*> perDeviceIdleCV;
extern vector<bool> perDeviceIdleFlag;

extern  map<string, mutex*> ReqMtxVec;
extern vector<mutex*> ReqCmpMtx;
extern  map<string, mutex> perTaskWTMtx;
extern vector<shared_ptr<condition_variable>> ReqCmpCV;

extern map<string, int> perTaskCnt;

//Newly added 2019-7-19 : need tables to track whether a model uses single definition for all devices or not.
// in
extern map<string, string> reqNametoNetNameforCPU; 
extern map<string, string> reqNametoNetNameforGPU; 

extern int batchTaskID;
extern mutex batchMtx;

//MPS related tables
vector<int> MPSTaskcnt={0,0,0,0};

int  MPSSendCnt=0;
mutex cntMtx;

int MPSRecvCnt=0;
mutex cntMtx2;


void* sendandWait(shared_ptr<batched_request> input_info, string strReqName, shared_ptr<TaskSpec> pTask, proxy_info* pPInfo);
void ProxyBasicSetup(proxy_info* pPInfo);

pthread_t initProxyThread(proxy_info* pPInfo){
    mutex randnumMtx; // use lock in order to ensure unique numberint deviceID = gpu_id;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024); // set memory size, may need to adjust in the future for now set to max * 1MB
    pthread_t tid;
    if (pthread_create(&tid, &attr, initProxy, (void *)pPInfo) != 0)
        LOG(ERROR) << "Failed to create a batching thread.\n";
    return tid;

}

//// check list, batch requests as much as possible  and call exeuction handler ////
void*  initProxy(void *args){
    proxy_info* pPInfo = (proxy_info*)args;
    int DeviceID = pPInfo->dev_id;
    int cap = pPInfo->cap;
    int dedup_num = pPInfo->dedup_num;
    DeviceSpec *ptemp = new DeviceSpec;
    int GPUProxy_in; //fd for sending input to gpu Proxy
    int GPUProxy_out; //fd for receiving output from gpu Proxy

    initMtx.lock();
    
    if (gsc.isGPU(DeviceID)){ // if GPU
#ifdef FIXED_CORES
        cpu_set_t cpuset;
        pthread_t thread=pthread_self();
        for (int i =0; i< 2; i++){
        CPU_SET(gpucores[DeviceID][i], &cpuset);
         printf("core %d set for GPU %d\n",gpucores[DeviceID][i],DeviceID);
        }
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)!=0){
            printf("Error in setting affinity for cpu thread\n");
            exit(1);
        }

#endif
        ptemp->type="gpu";
         }
    else {// if CPU
        ptemp->type="cpu";
        cpu_set_t cpuset;
        pthread_t thread=pthread_self();
#if NUMA_AWARE 
        int CPUID;
        if (numaID < MAX_NUMA_NODE)
            CPUID = 0;
        else
            CPUID = 1;
        numaID++;
        CPU_ZERO(&cpuset);
        const int perCPUCores = TOTAL_CORES/MAX_NUMA_NODE;
        for (int i = perCPUCores*CPUID; i<perCPUCores+perCPUCores*CPUID; i++){ // this way may not work for other machines, must check each core's numa node in the future 
            CPU_SET(i, &cpuset);
        }
        int s;
        s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        if (s!=0){
            printf("Error in setting affinity for cpu thread\n");
            exit(1);
        }
        setNumaMemoryStack(perCPUCores*CPUID);// must guarantee every(including first) core belongs to the same node
   #else
#ifdef FIXED_CORES
        int cpuID = DeviceID - gsc.getNGPUs();
        const int perCPUCores = sizeof(cpucores)/sizeof(int)/gsc.getNCPUs();
        for (int i = perCPUCores*cpuID; i<perCPUCores+perCPUCores*cpuID; i++){
            CPU_SET(cpucores[i], &cpuset);
            printf("core %d set for CPU %d\n",cpucores[i],cpuID);
        }
        if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)!=0){
            printf("Error in setting affinity for cpu thread\n");
            exit(1);
        }
        ptemp->numcores = perCPUCores;

    #else
        int cpuID = DeviceID - gsc.getNGPUs();
        const int perCPUCores = gsc.getMaxCores()/gsc.getNCPUs();
        for (int i = perCPUCores*cpuID; i<perCPUCores+perCPUCores*cpuID; i++){
            CPU_SET(i, &cpuset);
            printf("core %d set for CPU %d\n",i,cpuID);
        }
        if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)!=0){
            printf("Error in setting affinity for cpu thread\n");
            exit(1);
        }
        ptemp->numcores = perCPUCores;

#endif
#endif

    }
#ifdef DEBUG
    printf("perProxyIsSetup[DeviceID: %d,cap: %d]: %s\n",DeviceID,cap,pPInfo->isConnected ? "true":"false");
#endif 
    if (!pPInfo->isConnected){ // in order to avoid duplicate setup we check first!
        
        ProxyBasicSetup(pPInfo);//set up server
        ServerState.DeviceSpecs.push_back(ptemp);
        pPInfo->isConnected=true;
    }


    initMtx.unlock();
    uint64_t start,end;
    while (1){
        //wait for condition variable to be called
        
        deque<shared_ptr<TaskSpec>> *pBatchList = ServerState.perProxyBatchList[{pPInfo->dev_id, pPInfo->cap}];
        unique_lock<mutex> lk(*perProxyBatchMtx[{pPInfo->dev_id, pPInfo->cap}]); 
        perProxyBatchCV[{pPInfo->dev_id, pPInfo->cap}]->wait(lk, [&pBatchList]{return pBatchList->size();});
        shared_ptr<TaskSpec> task =  pBatchList->front();
        string StrReqName = task->ReqName;
        queue<shared_ptr<request>>* pReqList=&ServerState.ReqListHashTable[StrReqName];
#ifdef DEBUG
       printf("[BATCH][%d,%d] batch list size: %lu, front task : %s\n",pPInfo->dev_id, pPInfo->cap,pBatchList->size(),StrReqName.c_str());
#endif 
        if (pReqList->empty()) { // possible if previous thread has already popped most of the requests
#ifdef DEBUG
       printf("[BATCH][%d,%d] however, request list is empty, thus exiting \n",pPInfo->dev_id, pPInfo->cap);       
#endif 
            pBatchList->pop_front();
            lk.unlock();
            perProxyBatchCV[{pPInfo->dev_id, pPInfo->cap}]->notify_one();
            continue;
        }
        pBatchList->pop_front();
        lk.unlock();
        perProxyBatchCV[{pPInfo->dev_id,pPInfo->cap}]->notify_one();
        shared_ptr<batched_request> pBatchedReq= make_shared<batched_request>(); // the local batched_request;
        batchMtx.lock();
        pBatchedReq->setBatchID(batchTaskID++);
        batchMtx.unlock();
        pBatchedReq->setReqName(StrReqName.c_str());
        // 1 set the task's device which it will run on
        pBatchedReq->setGPUID(DeviceID);
        bool doSkip=false;
        ReqMtxVec[StrReqName]->lock();
        if (pReqList->empty()) doSkip = true;
        ReqMtxVec[StrReqName]->unlock();
        if (doSkip){ // if empty queue, no need to wait, this part exists to make sure it does not work for empty queue
            //delete pBatchedReq;
            pBatchedReq.reset();
            perProxyBatchCV[{pPInfo->dev_id, pPInfo->cap}]->notify_one();
#ifdef DEBUG
            printf("[BATCH][%d,%d] request list is empty some time later, thus exiting \n",pPInfo->dev_id, pPInfo->cap);  
#endif 
            continue;
        }
        if (perTaskisWaiting[StrReqName]) {// might happen if serveral devices decided to wait for a task 
            //delete pBatchedReq;
            pBatchedReq.reset();
            perProxyBatchCV[{pPInfo->dev_id, pPInfo->cap}]->notify_one();
#ifdef DEBUG
            printf("[BATCH][%d,%d] task %s is being processed already, thus exiting \n",pPInfo->dev_id, pPInfo->cap,StrReqName.c_str());  
#endif 
            continue;
        }
        perTaskisWaiting[StrReqName] = true;
         perProxyExecMtx[{pPInfo->dev_id,pPInfo->cap}]->lock();
        ServerState.perProxyExecutingList[{pPInfo->dev_id, pPInfo->cap}]++;
        perProxyExecMtx[{pPInfo->dev_id,pPInfo->cap}]->unlock();

        int maxbatch;
        int maxdelay;
        // 2 decide the maximum batch size allowed 
        maxbatch = gsc.getMaxBatch(StrReqName,"gpu");
        maxdelay = gsc.getMaxDelay(StrReqName,"gpu", pReqList->size());
        if(task->BatchSize != -1) pBatchedReq->setMaxBatch(task->BatchSize);
        else pBatchedReq->setMaxBatch(maxbatch);
        
        usleep(maxdelay * 1000);
        perTaskisWaiting[StrReqName] = false;

        perTaskBatchingMtx[StrReqName]->lock();
        while(pBatchedReq->getBatchNum() < pBatchedReq->getMaxBatch())
        {
            ReqMtxVec[StrReqName]->lock();
            if (pReqList->empty()){
                    ReqMtxVec[StrReqName]->unlock();
                    break;
            }
            shared_ptr<request> r=  pReqList->front();
            pReqList->pop();
            ReqMtxVec[StrReqName]->unlock();
            r->setDeviceID(DeviceID);
            pBatchedReq->addRequestInfo( r, gsc.getNumOfInputTensors(StrReqName));
            ServerState.WaitingTable[StrReqName]--;
        }
        perTaskBatchingMtx[StrReqName]->unlock();
        if(pBatchedReq->b_tensors.size() !=0){ 
           for (int id=0; id< pBatchedReq->getNTask();id++){
                pBatchedReq->taskRequests[id]->setendBatch(getCurNs());
            } 
            sendandWait(pBatchedReq,StrReqName,task,pPInfo);
            //schedMtx.lock();
            //gsc.doScheduling(&ServerState);
            //schedMtx.unlock();
        }
        else{
#ifdef DEBUG
            printf("[BATCH][%d,%d] no requests found for task %s in the end, thus exiting \n",pPInfo->dev_id, pPInfo->cap,StrReqName.c_str());  
#endif 
            perProxyExecMtx[{pPInfo->dev_id,pPInfo->cap}]->lock();
            ServerState.perProxyExecutingList[{pPInfo->dev_id, pPInfo->cap}]--;  
            perProxyExecMtx[{pPInfo->dev_id,pPInfo->cap}]->unlock();
        }
        pBatchedReq.reset();
    }// infinite loop, for perProxyBatchCV
}//batch handler function

void ProxyBasicSetup(proxy_info* pPInfo){
#ifdef PYTORCH
    std::map<string,shared_ptr<torch::jit::script::Module>> *tlnets = new map<string, shared_ptr<torch::jit::script::Module>>();
#endif 
    
    int GPUProxy_in=connectGPUProxyIn(pPInfo->dev_id, pPInfo->cap,pPInfo->dedup_num);
    int GPUProxy_out=connectGPUProxyOut(pPInfo->dev_id,pPInfo->cap,pPInfo->dedup_num);
#ifdef DEBUG
    printf("Device %d. inFD: %d, outFD; %d\n",pPInfo->dev_id,GPUProxy_in,GPUProxy_out);
#endif 
    pPInfo->in_fd=GPUProxy_in;
    pPInfo->out_fd=GPUProxy_out;
    pPInfo->isConnected=true;
    int deviceID = pPInfo->dev_id;
    pPInfo->sendMtx = new mutex();
// Initiate locks and CVs
    if(!pPInfo->dedup_num){ // skip if the proxy is a duplicate 
        perProxyBatchMtx[{pPInfo->dev_id, pPInfo->cap}]=new mutex();
        perProxyExecMtx[{pPInfo->dev_id, pPInfo->cap}]=new mutex();

        condition_variable *pCV = new condition_variable();
        perProxyBatchCV[{pPInfo->dev_id, pPInfo->cap}]=pCV;

        ServerState.perProxyBatchList[{pPInfo->dev_id, pPInfo->cap}]=new deque<shared_ptr<TaskSpec>>();
        ServerState.perProxyExecutingTask[{pPInfo->dev_id, pPInfo->cap}]=new deque<shared_ptr<TaskSpec>>();
        ServerState.perProxyExecutingList[{pPInfo->dev_id, pPInfo->cap}]=0;
        pPInfo->isSetup=true;
    #ifdef DEV_LOG
    string proxyname = "proxy_log";
    proxyname = proxyname + to_string(pPInfo->dev_id)+to_string(pPInfo->cap);
    proxyname = proxyname + ".txt";
    devlogTable[{pPInfo->dev_id,pPInfo->cap}]=fopen(proxyname.c_str(), "w");
    #endif 
    }

// Initiate scheduler related variables
    if (!gsc.isGPU(deviceID)){//init cpu server
        ServerState.DeviceMaxScheduled[deviceID]=MAX_CPU_TYPE;
        ServerState.DeviceMaxRun[deviceID]=MAX_CPU_RUN_SLOT;

    }
    else if(gsc.isGPU(deviceID)){  //init gpu Server
        ServerState.DeviceMaxScheduled[deviceID]=MAX_GPU_TYPE;
        ServerState.DeviceMaxRun[deviceID]=MAX_GPU_RUN_SLOT_MPS;
    }   

// Load model weights
    ifstream file(NET_LIST.c_str());
    string NetName;
    string dev;
    while (file >> NetName){
        string delimiter = "-";
        string token = NetName.substr(0, NetName.find(delimiter));
        string dev;
        string reqname = token;
        if (!USE_MPS){
#ifdef PYTORCH
                string ptfile = COMMON + "/models/" + NetName+".pt";
                (*tlnets)[NetName] = torch::jit::load(ptfile.c_str());
                if (gsc.isGPU(deviceID)) { // if loading to GPU 
                    torch::Device gpu_dev(torch::kCUDA,deviceID);
                    (*tlnets)[NetName] -> to(gpu_dev);
                }
                else{
                    torch::Device cpu_dev(torch::kCPU);
                     (*tlnets)[NetName] -> to(cpu_dev);
                }
#endif 
        }      
       // init net hashes 
        //first check whether is a list for the hash
        map<string,queue<request>>::iterator it;
        if (ServerState.ReqListHashTable.find(reqname) == ServerState.ReqListHashTable.end()){
#ifdef DEBUG 
            printf("inserting %s into hash table(s)\n", reqname.c_str());
#endif
            perTaskCnt[reqname]=0; // initiate pertask count
            queue<shared_ptr<request>> *rinput = new queue<shared_ptr<request>>;
            ServerState.ReqListHashTable[reqname]=*rinput;
            //also make a new mutex for the request
            ReqMtxVec[reqname] = new mutex();
            perTaskBatchingMtx[reqname] = new mutex();
            perTaskisWaiting[reqname] = false;

            // complete queues and corresponding mutexes
            cmpTableMtx[reqname]=new mutex();
            
        }

        if (gsc.isGPU(deviceID)){
            if (NetName != reqname){ dev =  NetName.substr(NetName.find(delimiter)+1, NetName.length());
                if (dev.find("gpu") == string::npos) continue;
            }
            cout << "settingup model " << NetName << " for GPU " <<endl;
            reqNametoNetNameforGPU[reqname]=NetName;
               
        }
        else{
            if (NetName != reqname) {
                dev =  NetName.substr(NetName.find(delimiter)+1, NetName.length());
                if (dev.find("cpu") == string::npos) continue;
            }
            cout << "settingup model " << NetName << " for CPU " <<endl;
            reqNametoNetNameforCPU[reqname]=NetName;
        }
        // initiate shared tables that will be referenced by other threads
        ServerState.WaitingTable[reqname]=0;
        ServerState.PerfTable[reqname]=0;
        ServerState.perTaskGenCnt[reqname]=0;
        cout << "preloaded "<<NetName<<endl;
    }
    ServerState.Nets.push_back(*tlnets);
}


void* sendandWait(shared_ptr<batched_request> input_info, string strReqName, shared_ptr<TaskSpec> pTask, proxy_info* pPInfo) {
    int gpu_id = input_info->getGPUID();
    char req_name[MAX_REQ_SIZE];
    string netName;
    netName = reqNametoNetNameforGPU[strReqName]; 
    strcpy(req_name, netName.c_str());
#ifdef PYTORCH
    map<string, shared_ptr<torch::jit::script::Module>>::iterator it = ServerState.Nets[gpu_id].find(req_name);
#endif 
        
    int tnum = input_info->getNTask();
#ifdef DEBUG
    printf("[EXEC][%d,%d,]there are total %d requests batched for task %s\n",pPInfo->dev_id,pPInfo->cap,tnum,req_name);
    printf("[EXEC][%d,%d]] batch ID : %d, batched task's task ID : ",pPInfo->dev_id,pPInfo->cap,input_info->getBatchID());
    for (int id=0; id<tnum;id++){
        printf("%d ",input_info->taskRequests[id]->getTaskID());
    }    
    printf("\n");
#endif 
    //vector<torch::Tensor> inputs;

    torch::Tensor batched_input = torch::cat(input_info->b_tensors,0);
      
    //inputs.push_back(batched_input);
    float *raw_data=batched_input.data<float>();
    float *dummy_output;
    int *dummy_len;
    pPInfo->sendMtx->lock();
 #ifdef DEV_LOG
    string msg=str_START + to_string(tnum);
    printTimeStampWithName(req_name,msg.c_str(),devlogTable[{pPInfo->dev_id,pPInfo->cap}]);
#endif 

    ServerState.perProxyExecutingTask[{pPInfo->dev_id,pPInfo->cap}]->push_back(pTask);
    for (int id=0; id<input_info->getNTask();id++){
        input_info->taskRequests[id]->setstartExec(getCurNs());
    } 
    int send_size = sendRequest(pPInfo->in_fd, req_name,input_info->getBatchID() ,batched_input);
    // recv 
    int ret_req_id=recvResult(pPInfo->out_fd,&dummy_output,dummy_len);
        pPInfo->sendMtx->unlock();
        //free(dummy_output);
        //free(dummy_len);
    
#ifdef DEV_LOG
    msg=str_END + to_string(tnum);
    printTimeStampWithName(req_name,msg.c_str(),devlogTable[{pPInfo->dev_id,pPInfo->cap}]);
    fflush(devlogTable[{pPInfo->dev_id,pPInfo->cap}]);
#endif 
     ServerState.isRunningVec[gpu_id]--;
    for (int id=0; id<tnum;id++){
        input_info->taskRequests[id]->setendExec(getCurNs());
    }
#ifdef DEBUG
    printf("[EXEC][%d,%d] finished task %s \n", pPInfo->dev_id,pPInfo->cap,req_name);
#endif 
    perProxyExecMtx[{pPInfo->dev_id,pPInfo->cap}]->lock();
    ServerState.perProxyExecutingTask[{pPInfo->dev_id,pPInfo->cap}]->pop_front();
    ServerState.perProxyExecutingList[{pPInfo->dev_id, pPInfo->cap}]--;  
    perProxyExecMtx[{pPInfo->dev_id,pPInfo->cap}]->unlock();
    ServerState.isRunningVec[gpu_id]--;
    sendBatchedResults(input_info, strReqName); 
    //ServerState.perDeviceBatchList[i]->pop_front();

}


