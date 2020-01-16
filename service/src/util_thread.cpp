#include <condition_variable>

#include "state.h"
#include "thread.h"
#include "mon.h"
#include "scheduler.h"

#define MUTRACE 1
using namespace std;
extern int TOTAL_CMPQUEUE;
extern GlobalScheduler gsc;
extern SysInfo ServerState;

extern bool utilstarted;
extern bool perfstarted;
extern bool EXIT;

extern condition_variable utilMonCV;
extern condition_variable perfMonCV;


pthread_t initGPUUtilThread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024); // set memory size, may need to adjust in the future, for now set it to 1MB
    //init batch index table
    pthread_t tid;
    if (pthread_create(&tid, &attr, GPUUtilMon, NULL) != 0)
        LOG(ERROR) << "Failed to create performance monitor thread.\n";
    return tid;

}


//the following function is deprecated but stil leave it for reference
void* GPUUtilMon(void * vp){
    int useconds = 50* 1000; // for every 100 ms
    mutex utilMtx;
    int ngpus=gsc.getNGPUs();
  vector<FILE*> utilFiles;
    for(int i =0; i < ngpus; i++){
        string devname="gpu_log";
        devname=devname+to_string(i);
        devname=devname+".txt";
        utilFiles.push_back(fopen(devname.c_str(),"w"));
    }

    unique_lock<mutex> lk(utilMtx);
  
    //initiate log files
    utilMonCV.wait(lk,[]{return utilstarted;});

    while(1){
    for (int id =0; id < ngpus; id++){
        getUtilization(*(gsc.getGPUMon(id)), ServerState.arrGPUUtil[id]);
        float util=ServerState.arrGPUUtil[id][3];
        fprintf(utilFiles[id],"%f\n",util);
        fflush(utilFiles[id]);
    }
    usleep(useconds);
    }
    lk.unlock();
    return (void*)0;
}

pthread_t initPerfMonThread(SysInfo* ServerState){ // for initiating performance monitor thread
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024); // set memory size, may need to adjust in the future, for now set it to 1MB
    //init batch index table
    pthread_t tid;
    if (pthread_create(&tid, &attr, PerfMon, (void*)ServerState) != 0)
        LOG(ERROR) << "Failed to create performance monitor thread.\n";
    return tid;
}

void* PerfMon(void *vp){ // internall excepts SysInfo* as input

#ifdef DEBUG
  //  printf("[PERFMON] All performance will be logged per 50ms to service/throughput_log.txt and queue_log.txt \n");
#endif
	SysInfo *ServerState = (SysInfo*)vp;
    FILE* ptpLog=fopen("throughput_log.txt","w");
    FILE* pqLog =fopen("queue_log.txt", "w");
    FILE* cmpqLog=fopen("cmp_queue_size.txt","w");
    int useconds = 10 * 1000;
    fprintf(ptpLog,"task,# of tasks executed / %dms \n",useconds/1000);
    fflush(ptpLog);
    fprintf(pqLog, "task,# of queued tasks / %dms \n",useconds/1000);
    fflush(pqLog);
    fprintf(cmpqLog, "# of queued tasks / %dms \n",useconds/1000);
    fflush(cmpqLog);
    string TaskName;
    int preExitCnt=0;
    mutex PerfMtx;
    unique_lock<mutex> lk(PerfMtx);
    perfMonCV.wait(lk,[]{return perfstarted;});
    while(1){
        bool bReqFound = false;
        for(map<string, int>::iterator it=ServerState->WaitingTable.begin(); it != ServerState->WaitingTable.end(); it++){
                bReqFound=true;
                TaskName = it->first;
                fprintf(pqLog,"%s,%lu \n",TaskName.c_str(),ServerState->ReqListHashTable[TaskName].size());
                fprintf(ptpLog,"%s,%d \n",TaskName.c_str(),ServerState->PerfTable[TaskName]);
                fflush(pqLog);
                fflush(ptpLog);
                ServerState->PerfTable[TaskName]=0;
        }

        fprintf(cmpqLog, "%lu \n",ServerState->cmpQ.size_approx());
        fflush(cmpqLog);

        if (EXIT) break;
        usleep(useconds);   
    }
    lk.unlock();
    return (void*)0; 
}

