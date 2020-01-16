#ifndef SCHED_H
#define SCHED_H

#include <deque>
#include <vector>
#include <queue>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <nvml.h>
#include <map>
#include <boost/lockfree/queue.hpp>

#include "tonic.h"
#include "ts_list.h"
#include "state.h"
#include "request.h"
#include "table.h" // for table based performance model 
#include "batched_request.h"
#include "concurrentqueue.h"
#include "gpu_proxy.h"
#include "interference_model.h"

using namespace std;
 enum scheduler {NO, STATIC, DYNAMIC ,FIFO,RR,RANDOM,MPS,GANG,MPS_TEST, SLO, WRR, WRR_VGPU};
typedef struct {
float compUtil;
float memUsage;
} ProcInfo;

typedef struct _DeviceSpec{
    string type; // stores either "cpu" or "gpu" for now
    // CPU specific variables
    int numcores; // stores the number of cores it has been allcoated
    uint64_t recentTimeStamp; // the most recent task's timestamp which has been sheduled for this device 
} DeviceSpec;

typedef struct _TaskSpec{
    int DeviceId;
    string ReqName;
    int BatchSize;
    int CapSize; // maximum amount of cap for this task, used in MPS environment
} TaskSpec; // mostly used for scheduling

typedef struct _ReqGenStruct{
    char StrReqName[MAX_REQ_SIZE];
    float rand_mean;
    int nrequests;
}ReqGenStruct;


//struct which stores system state related info, used in stateful scheduling
typedef struct _SysInfo{
	vector<map<string, shared_ptr<torch::jit::script::Module>>> Nets;
	map<string, queue<shared_ptr<request>>> ReqListHashTable; // per request type queue
    vector<queue<shared_ptr<request>>> CmpListVec; // per request type queue
    moodycamel::ConcurrentQueue<shared_ptr<request>> cmpQ;
    map<string, int> CompletedTable;
	map<string, int> WaitingTable; // used for keeping track of waiting requests in the queue, inc when accepted by socket, dec when batched
	map<string, int> PerfTable; // used for measuring throughput, checks how many requests have been sent back to client,refreshed when reported
	vector<int> isRunningVec; // vector of ints per device, tracks hwo many requests are to be batched/running on the GPU
//	map<int, ExecMode> DeviceExecMode; // table that stores which execution mode a device e running e.g) EB, CON, B2B
	map<int, int> DeviceMaxScheduled; // table for storing how many tasks can be scheduled on each device 
    map<int, int> DeviceMaxRun; // table for storing hwo many tasks can be running on each device 
    vector<float>GPUUtil; // array for storing GPU utilization, for each device, updated by calling updateGPUUtil(  )
    vector<DeviceSpec*> DeviceSpecs; // vector storing device specs of the system
    int NUM_CORES;
    vector<deque<shared_ptr<TaskSpec>>*> perDeviceBatchList; // list of tasks to batch
    map<pair<int,int>, deque<shared_ptr<TaskSpec>>*> perProxyBatchList; // list of tasks to batch for each [dev,cap] pair
    map<pair<int,int>, int> perProxyExecutingList; // number of tasks that are exeucting
    map<pair<int,int>, deque<shared_ptr<TaskSpec>>*> perProxyExecutingTask; // spec of task  that is send/waiting
    vector<int> perDeviceToExecList; // # of tasks that are waiting(to batch and execute)
    vector<deque<string>*> perDeviceExecutingList; // list of tasks executing
    float arrGPUUtil[4][4];
    vector<bool> perDeviceReserveFlag; 
    map<string, int> perTaskGenCnt;
    map<int, vector<int>> perDevMPSCap;
    map<int, vector<proxy_info*>> perDevMPSInfo;
} SysInfo;
class GlobalScheduler
{

public: 
GlobalScheduler();
~GlobalScheduler();


/*schedulers
 *all of the following returns int as a device ID 
 */


//below are methods related to scheduling
vector<shared_ptr<TaskSpec>> executeScheduler(SysInfo *SysState);
void doScheduling(SysInfo* SysState);
void doMPSScheduling(SysInfo* SysState);


// below are schedulers called according to _mode
vector<shared_ptr<TaskSpec>> noScheduler(SysInfo *SysState);
vector<shared_ptr<TaskSpec>> staticGreedy(SysInfo *SysState);
vector<shared_ptr<TaskSpec>> dynamicGreedy(SysInfo *SysState);
vector<shared_ptr<TaskSpec>> randomScheduler(SysInfo *SysState);
vector<shared_ptr<TaskSpec>> RRScheduler(SysInfo *SysState, bool weighted);
vector<shared_ptr<TaskSpec>> vGPUwRRScheduler(SysInfo *SysState, bool weighted);
vector<shared_ptr<TaskSpec>> FIFOScheduler(SysInfo *SysState);
vector<shared_ptr<TaskSpec>> gangScheduler(SysInfo *SysState);
vector<shared_ptr<TaskSpec>> staticMPSScheduler(SysInfo *SysState);
vector<shared_ptr<TaskSpec>> SLOScheduler(SysInfo *SysState);
vector<shared_ptr<TaskSpec>> creditScheduler(SysInfo *SysState);

int RandomScheduler(); // returns random number based on the number of GPUS

/*methods for setting up*/
void setupProcInfos(string dir_of_profile_file); // sets up profiling info by given directory
void setNGPUs(int nGPU);
void setMaxCPUCores(const int NCORES);
void setupMonitor();
int setSchedulingMode(string mode, bool isAdaptiveFlag);
void setupMaxBatchnDelay(string mb_file);
void setupMaxBatch(int maxBatch);
void setMaxBatch(string name, string type, int max_batch);
void setupMaxDelay(int maxDelay);
void setupNetNames(string net_file);
void setupNumCPUDevices(int cd);
void setupTableModel(string gpu_file, string cpu_file);
void setupOptCores(string optcore_file);
void setupWeightedEpoch(string we_file);
void setupWeights(string w_file);
void setupPriority(string pri_file, bool reverse);
void setupInputSpecs(string spec_file);
vector<ReqGenStruct> setupReqGenSpecs(string spec_file);
void setTaskisOpen(string TaskName, bool isOpen);
void setupMPSInfo(SysInfo *SysState, string capfile);
void setupSLO(string slofile);
void setupAvailvGPU(string specfile);
void setupEstModels();


void updateAvgReqInterval(string ReqName, float NewInterval); 

// misc useful methods called from server
bool isGPU(int deviceID);
void updateGPUUtil(SysInfo *SysState, int gpuid, string name, int batchsize, bool add); // gpuid : gpu ID, add(true)= addup, add(false) = subtrac
void resetGPUUtil(SysInfo *SysState, int gpuid); // resets back to 0
// epoch related methods
vector<string> checkEpochgetCandidate(vector<string> old_decision, int n);
void resetEpoch();

//credit related methods
void checkandRefreshCredit();
void initCreditTable();

void checkandRefreshCnt(SysInfo* SysState);

/*methods for getting numbers */
void printProfileInfo(); // for debuggung
int getMaxBatch(string name, string device); // get max number of batch, given request name and device type
int getMaxDelay(string reqname, string device, int QueueSize);
int getNGPUs();
int getNCPUs();
int getMaxCores();
int getNextIdleCPU(SysInfo *SysState);
bool getTaskisOpen(string TaskName);
float getTaskArivInterval(string TaskName);
djinn::TensorDataOption getDataOtption(string name);
int getNumOfInputTensors(string name);
double getEstimatedWaitExecTime(SysInfo *SysState, string ReqName, int deviceID);
double getEstimatedExecTime(SysInfo *SysState, string ReqName, int deviceID);
double getEstimatedBatchWaitTime(SysInfo *SysState, string ReqName, int deviceID);
double getEstLatency(string task, int batch, int cap);
int getMaxEstLatency(string task, int batch);
double getInterferedLatency(string mytask,int mybatch,int mycap,string cotask, int cobatch, int cocap);
map<string, int> getEstimatedBatchSize(SysInfo* SysState);
int getMPSCapID(shared_ptr<TaskSpec> pTask, SysInfo *SysState); // returns deviec cap for given parameters
int getWeightedRRCnt(string task, int batchsize);


nvmlDevice_t* getGPUMon(int id);
float getGPUUtil(string name, int batch);


private: 
scheduler _mode;
bool isBatchingAdaptive; // settedup when scheduling
map<string, ProcInfo*> Profile; // uses a "bench-batchsize" string as key and procInfo pointer as value
vector<string> net_names ; // vector containing names of network that can run on the server
int nGPUs; // number of GPUS
int nCPUs; // number of virtual CPU devices 
int UPPERBOUND; // used in grouped_fair scheduling
int MAX_CPU_CORES;
vector<nvmlDevice_t *> gpu_mons;
map<string,int> device_table; //table used for static schedulers that decide devices on server 
map<tuple<string, string>, int> mb_table;//max batch table 
map<tuple<string, string>, int> md_table; //max delay table
vector<float> sortedSpeedupVec; // used in greedy scheduling 
vector<string> sortedSpeedupTasks; // used in any speedup based scheduling, REALLY HANDY
map<string, bool> perTaskisOpen; // used in dynamic greedy scheduling
map<string, deque<float>> perTaskAvgArivInterval; // used in calculating delay 
map<string ,djinn::TensorDataOption> input_dataoption_table;
map<string ,int> input_datanum_table;

map<string, float> gpu_be_table; // used for min lat scheduling 
map<string, float> cpu_be_table; // used for min lat scheduling 
map<string, float> optcore_table; //used for min lat scheduling 
map<string, int> agingTable; // used in greedy dynamic and min_lat scheudling, initiated in setupnet_names
map<string, float> speedup_table; 
map<string, int> SLOTable; // used in SLO based scheduling
PerfTable tableModel;
PerfTable weightTable;
interference_modeling::interference_model model;

map<string, int> gpu_weightedEpochDelta;
map<string, int> cpu_weightedEpochDelta;

map<string, int> priority_table; 
map<string, int> gpu_credit_table; 
map<string, int> cpu_credit_table;
map<string, vector<int>> perTaskavailvGPU;
int GPU_TAU=600; //used in credit scheduler 
int CPU_TAU=2000; // used in credit scheduler 
int MA_WINDOW=10; //used for tracking moving average, this is the number of records to keep when getting average

int refresh_cnt=0;  // used in RR sceduler, for tracking how many times scheduling has occured 
int REFRESH_RATE=10; // used in RR scheduler, for deciding whether to refresh counters or not
int VGPU_QUEUE_CAP=10; //used in wrr_vgpu , used for capping batching tasks in queue
int vGPU_THRESHOLD=200; //used in wrr_vgpu, used for capping packed utilization
};

#endif 
