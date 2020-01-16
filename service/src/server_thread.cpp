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
#include "common_utils.h"
#include "cpuUsage.h"
#include "common_utils.h" //printTimeStamp moved to here
#include "client_model.h"

#include "input.h"
#include "timer.h"
#include "thread.h"
#include "ts_list.h"
#include "state.h"
#include "request.h"
#include "batched_request.h"
#include "scheduler.h"
#include "mon.h"

#define THREAD_PER_DEV 2
#define THREAD_PER_PROXY 2
#define SCHEDULING_INTERVAL 1
#define DUMMY_ACK 1
#define MUTRACE 0
const int VTHREAD_PER_DEV=10;
namespace po = boost::program_options; 
using namespace cv;
//using namespace std;


extern FILE* pLogFile;
extern bool USE_MPS;
extern bool EXP_DIST;
extern bool vGPU;

//2019-1-9 : just for checking 
map<string, int> perTaskCnt;


////system state related variables
SysInfo ServerState;

bool needScheduling;
bool CPU_UTIL=false;
bool HALF_PRECISON = false; // flag setted if we want to use half precision
bool WARMUP = true; // flag for indicating to do warmup
TonicSuiteApp app;
double rand_mean;
int nrequests;
int ncores;
int waitMillis;
extern GlobalScheduler gsc;

unsigned int taskID=0;

//  synch related variables       
std::map<std::string, mutex*> ReqMtxVec; // used when popping or pushing
std::map<std::string, mutex*> ReqExecMtxVec; // used to enforce only one thread to be batching a execution
std::mutex CntMtx; // mutex associated with task id counter value
std::mutex gscMtx; // "          global sheduler
std::mutex readyMtx;// "         ready flag
std::mutex clearMtx;// "             clear flag
std::mutex schedMtx;//   "           calling schedulers 

std::vector<mutex*> ReqCmpMtx; // mtx for  completed requests
std::vector<shared_ptr<condition_variable>> ReqCmpCV;

std::map<std::string, mutex*> cmpTableMtx;

std::mutex createMtx; // lock used when creating per task Mtxs
std::map<std::string, mutex*> perTaskArrivUpdateMtx; 
std::map<std::string, uint64_t> perTaskLastInterval;

map<std::string, std::mutex> perTaskWTMtx; // waiting table Mtx

std::condition_variable readyCV;
std::condition_variable completeCV;
std::mutex readyCVMtx; // the global_mutex used with the conditional variable
std::condition_variable perfMonCV;   
std::condition_variable utilMonCV;

/*Following flags are used in util_thread.cpp*/ 
bool perfstarted = false; //checks whether execution has started or not, for shorter logs!
bool utilstarted = false; //  //checks whether execution has started or not, for shorter logs!

std::map<string, int> perTaskDatalen; // used in recvTensor, need this prevent bad_alloc seg fault

bool wasIdle=false;

int ready=0; // works as a flag and semaphore
int clear=0; // works as a flag and semaphore 
int execCnt=0;
int complete=0; //works as a flag 


//std::queue<torch::Tensor> completeQueue;
unsigned int completedTasks; //no need to use a queue to just track bandwidth



//flag for start sending requests
bool startFlag;
bool finishedRequestsFlag; 

//condition variables and vector needed for batching threads
std::vector<condition_variable*> perDeviceBatchCV; 
std::vector<mutex*> perDeviceBatchMtx; 
std::vector<bool> perDeviceIsSetup;

//used for signaling idle device threads 
std::vector<condition_variable*> perDeviceIdleCV;
std::vector<bool> perDeviceIdleFlag;
                                             

pthread_t initRequestThread(int sock){                                                                                           
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024); // set memory size, may need to adjust in the future, for now set it to 1MB
    pthread_t tid;
    if (pthread_create(&tid, &attr, handleRequest, (void *)(intptr_t)sock) != 0)
        LOG(ERROR) << "Failed to create a request handler thread.\n";
    return tid;

}

int noteandReturnDatalen(string ReqName, int dlen){
    if(!perTaskDatalen[ReqName]){
        perTaskDatalen[ReqName] = dlen;
    }
    return perTaskDatalen[ReqName];
}
#ifdef PYTORCH

int recvTensor(int SockNum, char NetName[MAX_REQ_SIZE], shared_ptr<request> pReq){
    int reqID;
    pReq->setBackend(djinn::pytorch);
    string strReqName(NetName);
  //  char* dummy = (char *)malloc(sizeof(int));
  //  dummy[0]='o';
  //  dummy[1]='k';
  //  dummy[2]=0;
    //int dummy = 1;
    for (int i =0; i<gsc.getNumOfInputTensors(strReqName); i++){
        std::vector<int64_t> dims;
        int dimlen = SOCKET_rxsize(SockNum);
        //DUMMY_ACK
        //SOCKET_send(SockNum, dummy,sizeof(int),true);
        //SOCKET_txsize(SockNum,SockNum);

      
        //printf("phase 1 \n ");
        pReq->setStart(getCurNs());
        for(int i =0; i <dimlen; i++) {
            dims.push_back(SOCKET_rxsize(SockNum));
            //DUMMY_ACK
        //SOCKET_send(SockNum, dummy,sizeof(int),true);

        //printf("phase 2 \n ");
        }
        int Datalen = SOCKET_rxsize(SockNum);
    //DUMMY_ACK
        //SOCKET_send(SockNum, dummy,sizeof(int),true);

        Datalen = noteandReturnDatalen(strReqName, Datalen);  // trying to prevent segfault happening here
        int rcvd;
        if (gsc.getDataOtption(strReqName) == djinn::KFLOAT32){
        float *inData = (float *)malloc(Datalen * sizeof(float));
        rcvd = SOCKET_receive(SockNum, (char *)inData,Datalen * sizeof(float) ,true);
        //DUMMY ACK 
         //SOCKET_send(SockNum, dummy,sizeof(int),true);

        //printf("phase 3 \n ");

         torch::TensorOptions options(torch::kFloat32);
         pReq-> pushInputData(convert_rawdata_to_tensor(inData, dims, options));

        }
        else if(gsc.getDataOtption(strReqName) == djinn::KINT64){
          long *inData = (long *)malloc(Datalen * sizeof(long));
          rcvd = SOCKET_receive(SockNum, (char *)inData,Datalen * sizeof(long) ,true);
          torch::TensorOptions options(torch::kInt64);
          pReq -> pushInputData(convert_rawdata_to_tensor(inData, dims, options));       
        }
        if (rcvd == 0) {
            gsc.setTaskisOpen(strReqName, false);
            return 1;  // Client closed the socket    
        }
        reqID = SOCKET_rxsize(SockNum);

        //DUMMY_ACK
        //SOCKET_txsize(SockNum, -1);
        pReq->setendReq(getCurNs());
    }
    
    pReq->setReqName(NetName);
    pReq->setReqID(reqID);
#ifdef DEBUG
    printf("[RECV]received %s task of reqID : %d\n",NetName, reqID);
#endif
    //free(dummy);
    return 0;
}

#endif 

void* handleRequest(void* sock){
    int SockNum = (intptr_t)sock;
    char NetName[MAX_REQ_SIZE]; 
    //receive request 
    //printf("receiving name \n");
    SOCKET_receive(SockNum, (char*)&NetName, MAX_REQ_SIZE, false);
    //SOCKET_txsize(SockNum,-1);
    string StrName(NetName);
    perTaskDatalen[StrName]=0;
    gsc.setTaskisOpen(StrName, true);
    uint64_t CurrInterval=0;
    //need to check valid request name
   if ( ServerState.ReqListHashTable.find(NetName) == ServerState.ReqListHashTable.end()) {
        printf("task : %s not found \n", NetName);
        return (void*)1;
    } else
        LOG(ERROR) << "Task " << NetName << " forward pass.";
    createMtx.lock();
    if (perTaskArrivUpdateMtx.find(StrName) == perTaskArrivUpdateMtx.end()){
            perTaskArrivUpdateMtx[StrName] = new mutex();
        
    }
    createMtx.unlock();

    while (1) {
        shared_ptr<request> pReq = make_shared<request>(taskID, SockNum, 1);
        CntMtx.lock(); //lock counter
        taskID++;
        CntMtx.unlock(); // unlock counter
        if (MUTRACE){
            if(taskID >= 80000) exit(0);
        }
        pReq->setState(djinn::QUEUED);
#ifdef PYTORCH
        if (recvTensor(SockNum, NetName, pReq)) break;
#endif 
        //SOCKET_txsize(SockNum,-1);

        perTaskLastInterval[StrName]  =  CurrInterval;
        CurrInterval = getCurNs();
        if(perTaskLastInterval[StrName] != 0) {
                perTaskArrivUpdateMtx[StrName]->lock(); // checked by MUTRACE but low
                gsc.updateAvgReqInterval(StrName,float(CurrInterval - perTaskLastInterval[StrName])/1000000);
                perTaskArrivUpdateMtx[StrName]->unlock();

#ifdef DEBUG
            printf("Arrival interval of task %s : %f \n", NetName, gsc.getTaskArivInterval(StrName));
#endif 
        }

        ReqMtxVec[StrName]->lock(); // CHECKED BY MUTRACE (VERY HIGH)
        ServerState.ReqListHashTable[StrName].push(pReq);           
        ReqMtxVec[StrName]->unlock();
        //perTaskWTMtx[StrName].lock(); //CHECED BY MUTRACE but low
        ServerState.WaitingTable[StrName]++; 
        //perTaskWTMtx[StrName].unlock(); 
        if (!perfstarted){
            perfstarted=true;
            perfMonCV.notify_all();
        }
        //readyMtx.lock();
        ready++; // set the flag!
        //readyMtx.unlock();
        //readyCV.notify_one();   
        //delete pReq;
    }
    close(SockNum);
}  

pthread_t initServerThread(int numGPU){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8*1024 * 1024); // set memory size, may need to adjust in the future, for now set it to 1MB
    pthread_t tid;
    if (pthread_create(&tid, &attr, initServer, (void *)(intptr_t)numGPU) != 0)
        LOG(ERROR) << "Failed to create a request handler thread.\n";
    return tid;
}
void* initServer(void* numGPU){

    wasIdle=true;
    int _numGPU = (intptr_t)numGPU;
#ifdef USE_CPU
     for (int i =0; i<_numGPU + gsc.getNCPUs();i++) {  // need to initiate # of gpus + 1 settings, + number CPU devices

#else
     for (int i =0; i<_numGPU;i++) {  // need to initiate # of gpus + 1 settings, + number of virtual CPU devices
#endif
        ServerState.isRunningVec.push_back(0);

        if(!USE_MPS){
        // in order to avoid duplicate setups we set a per device flag
            perDeviceIsSetup.push_back(false);
        //init threads for each device
            if(!vGPU){
                for(int j =0; j<THREAD_PER_DEV;j++){
                    initDeviceThread(i);
                    sleep(1);
                }
            }
            else{

                for(int j =0; j<VTHREAD_PER_DEV;j++){
                    initDeviceThread(i);
                    sleep(1);
                }

            }
        }
        else{
            for(vector<proxy_info*>::iterator it = ServerState.perDevMPSInfo[i].begin(); it!=ServerState.perDevMPSInfo[i].end(); it++){
                for(int j =0; j<THREAD_PER_PROXY;j++){
                    initProxyThread(*it);
                    sleep(1);
                }
               //#ifdef DEBUG
                printf("initiated thread for dev: %d, cap: %d, dedup: %d \n", (*it)->dev_id,(*it)->cap,(*it)->dedup_num);
//#endif
            } 
            
        }
    }
    
    int cnt=0;        
    while (1){
        //monitor condition 
        //unique_lock<mutex> lk(readyCVMtx);  
#ifdef DEBUG 
       //printf("[SERV]server will wait on readyCV \n");
#endif

        //readyCV.wait(lk, []{return ready;}); // waits until there is a requests
        //

        while(!ready) {
            usleep(1000);
        }
 #ifdef DEBUG 
       //printf("[SERV]server finished waiting \n");
#endif
       
                
#ifdef DEBUG
            //uint64_t start,end;
            //start =getCurNs();
#endif 
            if (USE_MPS) gsc.doMPSScheduling(&ServerState);
            else gsc.doScheduling(&ServerState);
#ifdef DEBUG
            //end=getCurNs();
            //printf("[SCHEDULE] scheduling_latency(ms) : %lu \n", (end-start)/1000000 );
#endif 
          usleep(SCHEDULING_INTERVAL * 1000);
 
        // fail safe
        /*cnt =0;
         for (int i =0; i < gsc.getNGPUs() + gsc.getNCPUs(); i++){
            cnt+=ServerState.perDeviceBatchList[i]->size();
        }
#ifdef DEBUG
        //printf("[SERV] cnt : %d \n",cnt); 
#endif
        if (cnt == 0)
            wasIdle = true;
       if(wasIdle){
            wasIdle = false;*/
#ifdef DEBUG 
       //printf("[SERV]server will execute scheduler!(idle->non idle) \n");
#endif
        //    schedMtx.lock();
        //    gsc.doScheduling(&ServerState); 
        //    schedMtx.unlock();
        //readyMtx.lock();
        //ready=0;
        //readyMtx.unlock();


        //}
        

        //monitors ready queues and resets 'ready' if there are left overs in the queue
        //if(!ready){ // check in case there are still left-over requests in linked list
            for(map<string, queue<shared_ptr<request>>>::iterator it=ServerState.ReqListHashTable.begin(); it != ServerState.ReqListHashTable.end(); it++ ){
                if(it->second.size()){
                    readyMtx.lock();
                    ready=it->second.size();
                    needScheduling=true;
                    readyMtx.unlock();
                }
            }
            if (!ready){ // if still not ready, it means there are really no tasks in all queues
                wasIdle = true;
#ifdef DEBUG 
                printf("[SERV]server is turning idle! \n");
#endif
            }
       // }                                                                   
       // lk.unlock();
    } // infinite loop
    return (void*)1;
}


pthread_t initSendResultsThread(int id){
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8*1024 * 1024); // set memory size, may need to adjust in the future, for now set it to 1MB
        pthread_t tid;
        if (pthread_create(&tid, &attr, initSend, (void*)(intptr_t)id) != 0)
            LOG(ERROR) << "Failed to create a send results thread.\n";
        return tid;
}
void* initSend(void *args){
    //get front tasks from completedTasks
//    uint64_t loop_start,loop_end;
//    uint64_t p1, p2; 
    //string* reqname = (string*)args;
    int id = (intptr_t)args;
    while(true){
//    std::unique_lock<mutex> lk(*ReqCmpMtx[id]);
//    ReqCmpCV[id]->wait( lk,[id]{return !ServerState.CmpListVec[id].empty();}); // CHECKED by MUTRACE, moderate contenton #266 total 95.13, avg 0.001, max 13.977ms
#ifdef DEBUG
    //printf("[SEND_ACK] completeCV finished wait, %lu tasks in queue \n", ServerState.CmpListHashVec[id].size());
#endif 
    shared_ptr<request> tReq;
  //  ServerState.CmpListVec[id].pop();
    while(!ServerState.cmpQ.try_dequeue(tReq)){
        usleep(1*1000);

    }
 //   lk.unlock();
    //while(!ServerState.CmpListHashTable[*reqname].empty()){
    //    shared_ptr<request> tReq=ServerState.CmpListVec[id].front();

//    loop_start=getCurNs();
    //completeMtx.lock();
    /*    if(!tReq) {
        printf("AHH! \n");
//        ServerState.CmpListHashTable[*reqname].pop();
        break;
    }    */
        //ServerState.CmpListHashTable[*reqname].pop();
    //ServerState.completedTasks.pop();
    //completeMtx.unlock();
    //if (tReq == nullptr) continue;
    //    p1 = getCurNs();
//printf("[SEND_ACK] part1_latency : %f \n",float(p1-loop_start)/1000);
//    p1 = getCurNs();
 //   SOCKET_txsize(tReq->getClientFD(),tReq->getReqID());
//    p2 = getCurNs();
//printf("[SEND_ACK] send_latency : %f \n", float(p2-p1)/1000);
// p1 = getCurNs();
    tReq->setendSend(getCurNs());
    tReq->writeToLog(pLogFile);
    string StrName(tReq->getReqName());
    cmpTableMtx[StrName]->lock();
    ServerState.CompletedTable[StrName]++;  
    cmpTableMtx[StrName]->unlock();
    //ServerState.PerfTable[StrName]++;
    
    //tReq.reset();
//    loop_end=getCurNs();
 //   printf("[SEND_ACK] part2_latency : %f \n", float(loop_end-p1)/1000);
 //   printf("[SEND_ACK] sending_latency %f \n", float(loop_end-loop_start)/1000);

    //}// while (ServerState.completedTasks.empty())
    }//while(true)

}

pthread_t initFillQueueThread(ReqGenStruct *args){
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8*1024 * 1024); // set memory size, may need to adjust in the future, for now set it to 1MB
        pthread_t tid;
        if (pthread_create(&tid, &attr, initFill, (void*)args) != 0)
            LOG(ERROR) << "Failed to create a fill queue thread.\n";
        return tid;

}
void* initFill(void *args){
    ReqGenStruct* pReqArgs = (ReqGenStruct*)args;
    string StrName(pReqArgs->StrReqName);
    float rand_mean = pReqArgs->rand_mean;
    int nrequests = pReqArgs-> nrequests;
    // start filling data for n requests
    uint64_t CurrInterval=0;
    double rand_interval;
    client_model *pRandModel = new client_model(1,1/rand_mean);
    for(int i=0; i < nrequests; i++){
        if(!EXP_DIST)
            usleep(rand_mean* 1000 * 1000);
        else{
            rand_interval=pRandModel->randomExponentialInterval(rand_mean,1);
            usleep(rand_interval * 1000 * 1000);
        }
        CntMtx.lock(); //lock counter
        taskID++;
        CntMtx.unlock(); // unlock counter
        torch::Tensor input;
        shared_ptr<request> pReq = make_shared<request>(taskID, 0, 1); // * second parameter of request used to be SockNum, third is batchsize
        pReq->setStart(getCurNs());
        if( strcmp(pReqArgs->StrReqName,"dcgan") == 0 ) input=getRandLatVec(1);
        else input=getRandImgTensor();
        pReq->pushInputData(input);
        pReq->setReqName(pReqArgs->StrReqName);
        ReqMtxVec[StrName]->lock(); 
        ServerState.ReqListHashTable[StrName].push(pReq);       
        ReqMtxVec[StrName]->unlock(); 
        pReq->setendReq(getCurNs());
         ready++; // set the flag!
        readyCV.notify_one();

#ifdef DEBUG
         //printf("queue size of %s is %lu \n",pReqArgs->StrReqName,ServerState.ReqListHashTable[StrName].size());

#endif 
        ServerState.WaitingTable[StrName]++; 
        //perTaskWTMtx[StrName].unlock(); 
        if (!perfstarted){
            perfstarted=true;
            perfMonCV.notify_all();
        }
   	perTaskLastInterval[StrName]  =  CurrInterval;
        CurrInterval = getCurNs();
        if(perTaskLastInterval[StrName] != 0) {
                gsc.updateAvgReqInterval(StrName,float(CurrInterval - perTaskLastInterval[StrName])/1000000);
	}
#ifdef DEBUG
            printf("Arrival interval of task %s : %f \n", StrName.c_str(), gsc.getTaskArivInterval(StrName));
#endif 


    }
    return (void*)0;
}


