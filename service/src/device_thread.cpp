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
//2019-11-5 : added to print device exection LOG
vector<FILE*> devlogVec;

string str_START="START ";
string str_END="END ";
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
#define MAX_GPU_RUN_SLOT 1
#define MAX_GPU_RUN_SLOT_MPS 2
#define MAX_GPU_RUN_SLOT_VGPU 10
#define MAX_CPU_RUN_SLOT 1


// variables realted to main.cpp 
extern int TOTAL_CMPQUEUE;
extern GlobalScheduler gsc;
extern string NET_LIST;
extern string COMMON;
extern string WEIGHT_DIR;
extern bool needScheduling;
extern bool USE_MPS;
extern bool vGPU;
// exter variables realated to server_thread.cpp 
extern SysInfo ServerState;
extern int clear;
extern const int VTHREAD_PER_DEV;

//synch related extern variables
extern mutex schedMtx;
extern mutex clearMtx; 
extern condition_variable clearCV;
mutex initMtx;
mutex execMtx;

extern mutex completeMtx;
extern condition_variable completeCV;
extern map<std::string, mutex*> cmpTableMtx;;

//condition variables and vector needed for batching threads
extern vector<condition_variable*> perDeviceBatchCV;
extern vector<mutex*> perDeviceBatchMtx;
extern vector<bool> perDeviceIsSetup;
vector<mutex*> perDeviceExecMtx;

// following are synch variables for 
map<string, mutex*> perTaskBatchingMtx;
map<string, bool> perTaskisWaiting;


//used for signaling idle device threads 
extern vector<condition_variable*> perDeviceIdleCV;
extern vector<bool> perDeviceIdleFlag;

extern  map<string, mutex*> ReqMtxVec;
extern vector<mutex*> ReqCmpMtx;
extern  map<string, mutex> perTaskWTMtx;
extern vector<shared_ptr<condition_variable>> ReqCmpCV;

extern map<string, int> perTaskCnt;
extern int complete;

//Newly added 2019-7-19 : need tables to track whether a model uses single definition for all devices or not.
// in
map<string, string> reqNametoNetNameforCPU; 
map<string, string> reqNametoNetNameforGPU; 

int batchTaskID=0;
int queueID =0; //used in sendBatchedResults, for designating queues
mutex batchMtx;


// 
int clientSockRandNum=0;
mutex randnumMtx; // use lock in order to ensure unique number

void sendBatchedResults(shared_ptr<batched_request> brp, string reqname){
    uint64_t start,end;
    uint64_t s1,e1,s2,e2,s3,e3;
    int numofReq= brp->getNTask();
    for(int i =0; i < numofReq; i++){
    brp->taskRequests[i]->setendCmpq(getCurNs());
    ServerState.cmpQ.enqueue(brp->taskRequests[i]);
    }
    brp->taskRequests.erase(brp->taskRequests.begin(), brp->taskRequests.begin() + numofReq);
 #ifdef DEBUG
    //printf("pushed completed request \n");
#endif
        
}


void DeviceBasicSetup(int deviceID);

pthread_t initDeviceThread(int gpu_id){
    int clientSockRandNum=0;
    mutex randnumMtx; // use lock in order to ensure unique numberint deviceID = gpu_id;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024); // set memory size, may need to adjust in the future for now set to max * 1MB
    pthread_t tid;
    if (pthread_create(&tid, &attr, initDevice, (void *)(intptr_t)gpu_id) != 0)
        LOG(ERROR) << "Failed to create a batching thread.\n";
    return tid;

}

//// check list, batch requests as much as possible  and call exeuction handler ////
void*  initDevice(void *args){
    int DeviceID = (intptr_t)args;
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
    printf("perDeviceIsSetup[DeviceID: %d ]: %s\n",DeviceID,perDeviceIsSetup[DeviceID]? "true":"false");
#endif 
    if (!perDeviceIsSetup[DeviceID]){ // in order to avoid duplicate setup we check first!
        perDeviceIsSetup[DeviceID]=true;
        DeviceBasicSetup(DeviceID);//set up server
        ServerState.DeviceSpecs.push_back(ptemp);
    }
    if (!USE_MPS) warmupDevice(DeviceID); // if we use MPS proxy server will do the warmup


    initMtx.unlock();
    uint64_t start,end;
    while (1){
        //wait for condition variable to be called
        
        deque<shared_ptr<TaskSpec>> *pBatchList = ServerState.perDeviceBatchList[DeviceID];
        unique_lock<mutex> lk(*perDeviceBatchMtx[DeviceID]); 
        perDeviceBatchCV[DeviceID]->wait(lk, [&pBatchList]{return pBatchList->size();});
        shared_ptr<TaskSpec> task =  pBatchList->front();
        string StrReqName = task->ReqName;
        queue<shared_ptr<request>>* pReqList=&ServerState.ReqListHashTable[StrReqName];
#ifdef DEBUG
       printf("[BATCH] found %lu type of tasks to batch from device %d \n",pBatchList->size(),DeviceID);       
#endif 
        if (pReqList->empty()) { // possible if previous thread has already popped most of the requests
#ifdef DEBUG
       printf("[BATCH] However request list is empty, thus exiting \n");       
#endif 
            pBatchList->pop_front();
            lk.unlock();
            perDeviceBatchCV[DeviceID]->notify_one();
            continue;
        }
        pBatchList->pop_front();
        lk.unlock();
        perDeviceBatchCV[DeviceID]->notify_one();
        shared_ptr<batched_request> pBatchedReq= make_shared<batched_request>(); // the local batched_request;
        batchMtx.lock();
        pBatchedReq->setBatchID(batchTaskID++);
        batchMtx.unlock();
        //pBatchedReq = new batched_request();    
        pBatchedReq->setReqName(StrReqName.c_str());
        //pBatchedReq->setState(djinn::EMPTY);
        // 1 set the task's device which it will run on
        pBatchedReq->setGPUID(DeviceID);
        // 2 decide the maximum batch size allowed 
        bool doSkip=false;
        ReqMtxVec[StrReqName]->lock();
        if (pReqList->empty()) doSkip = true;
        ReqMtxVec[StrReqName]->unlock();
        if (doSkip){ // if empty queue, no need to wait, this part exists to make sure it does not work for empty queue
            //delete pBatchedReq;
            pBatchedReq.reset();
             perDeviceBatchCV[DeviceID]->notify_one();
            continue;
        }
        
        if (perTaskisWaiting[StrReqName]) {// might happen if serveral devices decided to wait for a task 
            //delete pBatchedReq;
            pBatchedReq.reset();
            perDeviceBatchCV[DeviceID]->notify_one();
            continue;
        }
       
        perTaskisWaiting[StrReqName] = true;
        perDeviceExecMtx[DeviceID]->lock();
        ServerState.perDeviceToExecList[DeviceID]++;
        perDeviceExecMtx[DeviceID]->unlock();

        int maxbatch;
        int maxdelay;
        if (gsc.isGPU(DeviceID) ){
                maxbatch = gsc.getMaxBatch(StrReqName,"gpu");
                maxdelay = gsc.getMaxDelay(StrReqName,"gpu", pReqList->size());
        }
        else{
                maxbatch = gsc.getMaxBatch(StrReqName,"cpu");
                maxdelay = gsc.getMaxDelay(StrReqName,"cpu", pReqList->size());
        }
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
            //perTaskWTMtx[StrReqName].lock(); 
            ServerState.WaitingTable[StrReqName]--;
            //perTaskWTMtx[StrReqName].unlock();
        }
        perTaskBatchingMtx[StrReqName]->unlock();
        if(pBatchedReq->b_tensors.size() !=0){ 
           for (int id=0; id< pBatchedReq->getNTask();id++){
                pBatchedReq->taskRequests[id]->setendBatch(getCurNs());
            } 
            while (ServerState.isRunningVec[DeviceID] >= ServerState.DeviceMaxRun[DeviceID]){
                usleep(1 *1000); //check every 1ms
            }
            ServerState.isRunningVec[DeviceID]++;
            handleExecution(pBatchedReq,StrReqName,task);
            //schedMtx.lock();
            //gsc.doScheduling(&ServerState);
            //schedMtx.unlock();
        }
        else{
            perDeviceExecMtx[DeviceID]->lock();
            ServerState.perDeviceToExecList[DeviceID]--;
            perDeviceExecMtx[DeviceID]->unlock();
        }
        pBatchedReq.reset();
    }// infinite loop, for perDeviceBatchCV
}//batch handler function

void DeviceBasicSetup(int deviceID){
#ifdef PYTORCH
    std::map<string,shared_ptr<torch::jit::script::Module>> *tlnets = new map<string, shared_ptr<torch::jit::script::Module>>();
#endif 
    
// Initiate locks and CVs
    perDeviceBatchMtx.push_back(new mutex());
    perDeviceExecMtx.push_back(new mutex());

    condition_variable *pCV = new condition_variable();
    perDeviceBatchCV.push_back(pCV);

    ServerState.perDeviceBatchList.push_back(new deque<shared_ptr<TaskSpec>>());
    ServerState.perDeviceToExecList.push_back(0);
    ServerState.perDeviceExecutingList.push_back(new deque<string>());
    if(vGPU) ServerState.GPUUtil.push_back(0);


// Initiate scheduler related variables
    if (!gsc.isGPU(deviceID)){//init cpu server
        ServerState.DeviceMaxScheduled[deviceID]=MAX_CPU_TYPE;
        ServerState.DeviceMaxRun[deviceID]=MAX_CPU_RUN_SLOT;

    }
    else if(gsc.isGPU(deviceID)){  //init gpu Server
        ServerState.DeviceMaxScheduled[deviceID]=MAX_GPU_TYPE;
        if(USE_MPS) ServerState.DeviceMaxRun[deviceID]=MAX_GPU_RUN_SLOT_MPS;
        else if (vGPU) ServerState.DeviceMaxRun[deviceID]=MAX_GPU_RUN_SLOT_VGPU;
        else ServerState.DeviceMaxRun[deviceID]=MAX_GPU_RUN_SLOT;
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
#ifdef DEV_LOG
    string devname = "dev_log";
    devname = devname + to_string(deviceID);
    devname = devname + ".txt";
    devlogVec.push_back(fopen(devname.c_str(), "w"));
#endif 
}


torch::Tensor SERVICE_fwd(shared_ptr<batched_request> batch,int deviceID, vector<torch::Tensor> input, int numoftypes ,shared_ptr<torch::jit::script::Module> module){
/*
    cpu_set_t cpuset;
    int corenum = (intptr_t)args;
    pthread_t thread=pthread_self();
    CPU_SET(corenum, &cpuset);
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)!=0){
        printf("Error in setting affinity for cpu thread\n");
    }        
*/
    std::vector<torch::jit::IValue> inputs;
    for (int i =0; i < numoftypes; i++){
        if(gsc.isGPU(deviceID)){
            torch::Device gpu_dev(torch::kCUDA,deviceID);
            input[i] = input[i].to(gpu_dev);
        }
        inputs.push_back(input[i]);
    }
    for (int id=0; id<batch->getNTask();id++){
        batch->taskRequests[id]->setstartExec(getCurNs());
    } 

    torch::Tensor output = module->forward(inputs).toTensor();
        // synch cuda calls
    if(gsc.isGPU(deviceID)) cudaDeviceSynchronize();
    for (int id=0; id<batch->getNTask();id++){
        batch->taskRequests[id]->setendExec(getCurNs());
    }

    return output;
}
void* handleExecution(shared_ptr<batched_request> input_info, string strReqName, shared_ptr<TaskSpec> pTask) {
    int gpu_id = input_info->getGPUID();
    char req_name[MAX_REQ_SIZE];
    string netName;
    if (gsc.isGPU(gpu_id))
        netName = reqNametoNetNameforGPU[strReqName]; 
    else        
        netName = reqNametoNetNameforCPU[strReqName]; 
    strcpy(req_name, netName.c_str());
#ifdef PYTORCH
    map<string, shared_ptr<torch::jit::script::Module>>::iterator it = ServerState.Nets[gpu_id].find(req_name);
#endif 
        
    int tnum = input_info->getNTask();
#ifdef DEBUG
    printf("[EXEC]there are total %d requests batched for task %s\n",tnum,req_name);
    printf("[EXEC] batch ID : %d, batched task's task ID : ",input_info->getBatchID());
    for (int id=0; id<tnum;id++){
        printf("%d ",input_info->taskRequests[id]->getTaskID());
    }    
    printf("\n");
#endif 
    vector<torch::Tensor> inputs;
    torch::Tensor batched_input = torch::cat(input_info->b_tensors,0);

    inputs.push_back(batched_input);
#ifdef DEV_LOG
    string msg=str_START + to_string(tnum);
    printTimeStampWithName(req_name,msg.c_str(),devlogVec[gpu_id]);
#endif 
        //uint64_t start_exec = getCurNs(); 
    //printf("START EXEC\n");
    perDeviceExecMtx[gpu_id]->lock();
    if(vGPU) gsc.updateGPUUtil(&ServerState, gpu_id, strReqName,pTask->BatchSize,true);
    perDeviceExecMtx[gpu_id]->unlock();
     torch::Tensor output = SERVICE_fwd(input_info, gpu_id,inputs,inputs.size(),ServerState.Nets[gpu_id][req_name]);
    //cudaDeviceSynchronize();
    //printf("END EXEC\n");

    //uint64_t end_exec = getCurNs();
    // printf("[EXEC] execution %f \n", float(end_exec-start_exec)/1000000);
#ifdef DEBUG
    printf("[EXEC] device %d finished %S \n",gpu_id, req_name);
#endif 
#ifdef DEV_LOG
    msg=str_END + to_string(tnum);
    printTimeStampWithName(req_name,msg.c_str(),devlogVec[gpu_id]);
    fprintf(devlogVec[gpu_id], "running tasks : %d\n",ServerState.isRunningVec[gpu_id]);
    fflush(devlogVec[gpu_id]);
#endif 
    ServerState.isRunningVec[gpu_id]--;
    perDeviceExecMtx[gpu_id]->lock();
    if(vGPU) gsc.updateGPUUtil(&ServerState, gpu_id, strReqName, pTask->BatchSize ,false);
    ServerState.perDeviceToExecList[gpu_id]--;
    perDeviceExecMtx[gpu_id]->unlock();

    //needScheduling=true;
    sendBatchedResults(input_info, strReqName); 

}

void warmupDevice(int devID){

#ifdef DEBUG
        printf("warming up on device %d\n",devID); 
#endif
#ifdef PYTORCH
        bool GPU = gsc.isGPU(devID);
        std::vector<torch::jit::IValue> inputs; 

        for (map<string,queue<shared_ptr<request>>>::iterator it = ServerState.ReqListHashTable.begin(); 
           it != ServerState.ReqListHashTable.end(); it++ ){
            torch::Tensor input;
           float* buffer;
            // form dummy data 
            if ( it->first == "vgg16" || it->first == "resnet18" || it->first == "alexnet" || it->first == "squeezenet"){
                std::vector<int64_t> dims={1,3,224,224};
                buffer = (float *)malloc(1*3*224*224*sizeof(float));
                memset(buffer, 0, 1*3*224*224*sizeof(float));
                torch::TensorOptions options(torch::kFloat32);
                input =  torch::from_blob(buffer, torch::IntList(dims), options);
            }
            else if (it->first == "dcgan"){
                std::vector<int64_t> dims={1,100,1,1};
                buffer = (float *)malloc(1*100*sizeof(float));
                memset(buffer, 0, 1*100*sizeof(float));
                torch::TensorOptions options(torch::kFloat32);
                input =  torch::from_blob(buffer, torch::IntList(dims), options);
            }
            if (GPU){
                torch::Device gpu_dev(torch::kCUDA,devID);
                input = input.to(gpu_dev);
            }  
            inputs.push_back(input);
            if (GPU){
                ServerState.Nets[devID][reqNametoNetNameforGPU[it->first]]->forward(inputs).toTensor();
                cudaDeviceSynchronize();
            }
            else ServerState.Nets[devID][reqNametoNetNameforCPU[it->first]]->forward(inputs).toTensor();
            inputs.clear();
            free(buffer);
        }

#endif 
#ifdef DEBUG
        printf("finished warming up\n");
#endif
    
}
