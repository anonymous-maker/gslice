
#include "state.h"
#include "scheduler.h"
#include "mon.h"
#include "common_utils.h"
#include "interference_model.h"
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <assert.h>


#define GPU_SLO 300
/*Below are all initialzied in initDevice thread */
extern vector<mutex*> perDeviceBatchMtx;
extern vector<condition_variable*> perDeviceBatchCV;
extern map<pair<int,int>, condition_variable*> perProxyBatchCV;
extern map<pair<int,int>,mutex*> perProxyBatchMtx;
extern map<string, mutex*> ReqMtxVec;
extern map<pair<int,int>,mutex*> perProxyExecMtx;
extern const int VTHREAD_PER_DEV;
extern bool USE_MPS;


GlobalScheduler::GlobalScheduler(){
nGPUs=0;
}

GlobalScheduler::~GlobalScheduler(){
//endNVML();
}

bool GlobalScheduler::isGPU(int deviceID){
        return deviceID < nGPUs;
}

int min_int(int a, int b){
    if (a<=b) return a;
    else return b;
}

map<string, int> GlobalScheduler::getEstimatedBatchSize(SysInfo* SysState){
    map<string, int> estimatedBatch;
    for (map<string,queue<shared_ptr<request>>>::iterator it = SysState->ReqListHashTable.begin(); it != SysState->ReqListHashTable.end(); it++){
        tuple<string, string> temp;
	    temp = make_tuple(it->first, "gpu");
        estimatedBatch[it->first] = min_int( mb_table[temp], it->second.size());
    }
    return estimatedBatch;
}

map<string, bool> getTasksNeedScheduling(SysInfo* SysState){
    map<string, bool> needDecision;
       for (map<string,queue<shared_ptr<request>>>::iterator it = SysState->ReqListHashTable.begin(); it != SysState->ReqListHashTable.end(); it++){
     	if(it->second.size()) {
            needDecision[it->first] = true;
#ifdef DEBUG
        
            printf("[SCHEDULER] task : %s needs scheduling \n", it->first.c_str());
        
#endif

        }
        else needDecision[it->first] = false;
    }
    return needDecision;
}

void GlobalScheduler::doScheduling(SysInfo* SysState){
#ifdef DEBUG
        printf("[SCHEUDLER] execute scheduler \n");
        //uint64_t startsched = getCurNs();
        for (int i =0; i < getNGPUs(); i++) 
        {                                                                                                                     
          printf("[SCHEDULER]BEFORE: batch: %d tasks, exec: %d tasks in Dev%d \n",SysState -> perDeviceBatchList[i]->size(),SysState->perDeviceToExecList[i], i);
        }
        bool didSchedule = false;        
#endif 
    	vector<shared_ptr<TaskSpec>> decision;
    	decision = executeScheduler(SysState);
#ifdef DEBUG
        for(int i =0; i < decision.size(); i++ ){
            printf("[SCHEDULER]AFTER: scheduled %s to device %d \n",decision[i]->ReqName.c_str(), decision[i]->DeviceId);
            didSchedule = true;
        }
        if (!didSchedule) printf("[SCHEDULER]AFTER: Nothing was scheduled  \n");
#endif
        if (decision.size()==0) return;
        
// force clear flag
        vector<bool> wasCleared;
#ifdef USE_CPU
        for(int i =0; i < nGPUs+nCPUs; i++) wasCleared.push_back(false);  
#else
           for(int i =0; i < nGPUs; i++) wasCleared.push_back(false);  

#endif
		for(int i =0; i < decision.size(); i++){
            int deviceID = decision[i]->DeviceId;
            perDeviceBatchMtx[deviceID]->lock();
            if(!wasCleared[deviceID]){ // force clear is required to enable forced priorities of task
                        //SysState -> perDeviceBatchList[deviceID]->clear(); 
                        wasCleared[deviceID]=true;
            }
		    SysState -> perDeviceBatchList[deviceID]->push_back(decision[i]);
            perDeviceBatchCV[deviceID]->notify_all(); 
            perDeviceBatchMtx[deviceID]->unlock();
        }   
}
void GlobalScheduler::doMPSScheduling(SysInfo* SysState){
#ifdef DEBUG
        printf("[SCHEUDLER] execute scheduler \n");
        //uint64_t startsched = getCurNs();
        
        for (int i =0; i < getNGPUs(); i++) 
        { 
            for (int j =0; j<SysState->perDevMPSInfo[i].size(); j++){
                proxy_info* pPInfo = SysState->perDevMPSInfo[i][j];
                printf("[SCHEDULER]BEFORE: batch: %lu tasks, exec: %d in proxy[%d,%d] \n",SysState-> perProxyBatchList[{pPInfo->dev_id,pPInfo->cap}]->size(),SysState->perProxyExecutingList[{pPInfo->dev_id, pPInfo->cap}], pPInfo->dev_id, pPInfo->cap);
            }
        }
        bool didSchedule = false;        
#endif 
    	vector<shared_ptr<TaskSpec>> decision;
    	decision = executeScheduler(SysState);
#ifdef DEBUG
        for(int i =0; i < decision.size(); i++ ){
            printf("[SCHEDULER]AFTER: scheduled %s to proxy %d , cap %d\n",decision[i]->ReqName.c_str(), decision[i]->DeviceId,decision[i]->CapSize);
            didSchedule = true;
        }
        if (!didSchedule) printf("[SCHEDULER]AFTER: Nothing was scheduled  \n");
#endif
        if (decision.size()==0) return;
        
   		for(int i =0; i < decision.size(); i++){
            int deviceID = decision[i]->DeviceId;
            int cap=decision[i]->CapSize;
            perProxyBatchMtx[{deviceID, cap}]->lock() ;
		    SysState -> perProxyBatchList[{deviceID,cap}]->push_back(decision[i]);
            perProxyBatchCV[{deviceID,cap}]->notify_all(); 
            perProxyBatchMtx[{deviceID,cap}]->unlock();
        
        }   
}



void GlobalScheduler::setMaxCPUCores(const int NCORES){
        MAX_CPU_CORES=NCORES;
}


nvmlDevice_t* GlobalScheduler::getGPUMon(int id){
    return gpu_mons[id]; 
}

void GlobalScheduler::setupNumCPUDevices(int cd){
    nCPUs = cd;
}


int GlobalScheduler::getNCPUs(){
    return nCPUs;
}

int GlobalScheduler::getMaxCores(){
    return MAX_CPU_CORES;
}

int GlobalScheduler::getNextIdleCPU(SysInfo * SysState){
    int idle_index=nGPUs; //suppose to return first cpu if not found , could change in the future
    for(int i = nGPUs; i< nGPUs+nCPUs; i++){
        if(SysState->perDeviceBatchList[i]->size() == 0){
            idle_index = i;
            break;
        }
    }
    return idle_index;
}
    
int GlobalScheduler::setSchedulingMode(string mode, bool isAdaptiveFlag){
    isBatchingAdaptive = isAdaptiveFlag;
	if (mode == "no"){
		_mode=NO;
	}
	else if(mode=="static"){
		_mode=STATIC;
	}
	else if(mode=="dynamic"){
		_mode=DYNAMIC;
	}
	else if(mode=="fifo"){
		_mode=FIFO;
	}
    else if(mode=="rr"){
        _mode=RR;
       }
    else if(mode =="random"){
        _mode=RANDOM;
       }
    else if(mode == "mps"){
        _mode=MPS;
    }
    else if(mode == "mps_test"){
        _mode=MPS_TEST;
    }
    else if(mode == "gang"){
        _mode=GANG;
    }
    else if(mode == "slo"){
        _mode=SLO;
    }
    else if(mode == "wrr"){
        _mode=WRR;
    }
    else if(mode == "wrr_sgpu")
        _mode=WRR_VGPU;
	else
		return 0;
    return 1;
		
}


void GlobalScheduler::setTaskisOpen(string TaskName, bool isOpen){
        perTaskisOpen[TaskName]=isOpen;

}
bool GlobalScheduler::getTaskisOpen(string TaskName){
        return perTaskisOpen[TaskName];

}
void GlobalScheduler::setupInputSpecs(string spec_file){
    ifstream infile(spec_file);
	string line;
	string token;
    bool first =true;
	//setup speedup table
	while(getline(infile,line)){
        if(first) {// used for skipping the first line
            first=false;
            continue;
        }
        
		istringstream ss(line);
		getline(ss,token,',');
		string name = token;
		getline(ss,token,',');
		string stroption = token;
        getline(ss,token,',');
        int numofdata = stof(token);

        if (stroption == "kFloat32"){
            input_dataoption_table[name] = djinn::KFLOAT32; // option in request.h
        }
        else if (stroption == "kInt64"){
            input_dataoption_table[name] = djinn::KINT64; // option in request.h
        }
        else {
            printf("No such data option : %s \n", stroption.c_str());
        } 
        input_datanum_table[name] = numofdata;
	//	printf("name : %s and speedup : %f\n", name.c_str(), speedup_table[name]);
	}
   
}


void GlobalScheduler::setupWeightedEpoch(string we_file){
    ifstream infile(we_file);
	string line;
	string token;
	//setup speedup table
	while(getline(infile,line)){
		istringstream ss(line);
		getline(ss,token,',');
		string name = token;
        getline(ss,token,',');
        string device = token; 
		getline(ss,token,',');
		int we = stof(token);
        if (device == "gpu")
		    gpu_weightedEpochDelta[name]=we;	
        else if (device == "cpu")
            cpu_weightedEpochDelta[name]=we;
	//	printf("name : %s and speedup : %f\n", name.c_str(), speedup_table[name]);
	}

}
void GlobalScheduler::setupWeights(string w_file){
    weightTable.createTableGPU(w_file);
#ifdef DEBUG
    weightTable.printTableContents();
#endif 
   
}
vector<ReqGenStruct> GlobalScheduler::setupReqGenSpecs(string spec_file){
    vector<ReqGenStruct> retVec;
     ifstream infile(spec_file);
	string line;
	string token;
	//setup speedup table
	while(getline(infile,line)){
        ReqGenStruct* pTemp = (ReqGenStruct*)malloc(sizeof(ReqGenStruct));
		istringstream ss(line);
		getline(ss,token,',');
		string reqname = token;
        strcpy (pTemp-> StrReqName,reqname.c_str());
        getline(ss,token,',');
        pTemp->rand_mean = stof(token); 
		getline(ss,token,',');
        pTemp->nrequests =stoi(token);
#ifdef DEBUG
	    printf("name : %s , mean : %f , nreq : %d \n", pTemp->StrReqName, pTemp->rand_mean, pTemp->nrequests);
#endif
        retVec.push_back(*pTemp);
	}
    return retVec;

}

void GlobalScheduler::setupTableModel(string gpu_file, string cpu_file){
    tableModel.createTableGPU(gpu_file);
    tableModel.createTableCPU(cpu_file);
#ifdef DEBUG
    tableModel.printTableContents();
#endif 
}

vector<shared_ptr<TaskSpec>> GlobalScheduler::executeScheduler(SysInfo *SysState){
    vector<shared_ptr<TaskSpec>> decision;
    switch(_mode){
                case NO:
                    decision = noScheduler(SysState);                
                    break;
                case RR:
                    decision= RRScheduler(SysState,false);
                    break;
                case STATIC:
                      decision = staticGreedy(SysState);
                    break;
                case FIFO:
                    decision=FIFOScheduler(SysState);
                    break;
                case DYNAMIC:
                    decision=dynamicGreedy(SysState);
                    break;
                 case MPS:
                    //map<int, string> temp_decision= mpsScheduler(SysState);
                    //break
                case GANG:
                    decision = gangScheduler(SysState);
                    break;
                case MPS_TEST:
                    decision = staticMPSScheduler(SysState);
                    break;
                case SLO:
                    decision = SLOScheduler(SysState);
                    break;
                case WRR:
                    decision= RRScheduler(SysState,true);
                    break;
                case WRR_VGPU:
                    decision= vGPUwRRScheduler(SysState, true);
                    break;
                defaut: // should not happen, conditions are already checked during initialization
                    break;
        }
   
    return decision;
}

void GlobalScheduler::setupMonitor(){
initNVML();
nvmlDevice_t *temp;
for (int i =0; i<nGPUs; i++){
        temp = (nvmlDevice_t*)malloc(sizeof(nvmlDevice_t));
        gpu_mons.push_back(temp);
        initMonitor(i,gpu_mons[i]);
}
}
bool cmpfunc_desc(float elem1, float elem2){
    return elem1> elem2; 
} 
bool cmpfunc_asc(float elem1, float elem2){
    return elem1< elem2; 
} 

bool cmpfunc_map_value_asc(pair<string, int> elem1 ,pair<string, int> elem2)
{
    return elem1.second < elem2.second;
                                
}
bool cmpfunc_map_value_desc(pair<string, int> elem1 ,pair<string, int> elem2)
{
    return elem1.second > elem2.second;
                                
}
bool cmpfunc_map_value_asc_uint64(pair<string, uint64_t> elem1 ,pair<string, uint64_t> elem2)
{
    return elem1.second < elem2.second;
                                
}
bool cmpfunc_map_value_desc_uint64(pair<string, uint64_t> elem1 ,pair<string, uint64_t> elem2)
{
    return elem1.second > elem2.second;
                                
}

void GlobalScheduler::updateAvgReqInterval(string ReqName, float NewInterval){
        perTaskAvgArivInterval[ReqName].push_back(NewInterval);
        if (perTaskAvgArivInterval[ReqName].size() > MA_WINDOW) perTaskAvgArivInterval[ReqName].pop_front();
}

float GlobalScheduler::getTaskArivInterval(string ReqName){

    //assert(perTaskAvgArivInterval[ReqName].size() !=0);
    if (perTaskAvgArivInterval[ReqName].size() ==0) // happens when first request is already batched and no intervals exist
        return 0;
    float sum=0;
    for(int i =0; i < perTaskAvgArivInterval[ReqName].size(); i++) sum += perTaskAvgArivInterval[ReqName][i];
    return sum / perTaskAvgArivInterval[ReqName].size() ; 
} 

void GlobalScheduler::setupPriority(string pri_file, bool reverse){
	ifstream infile(pri_file);
	string line;
	string token;
	//setup speedup table
	while(getline(infile,line)){
		istringstream ss(line);
		getline(ss,token,',');
		string name = token;
		getline(ss,token,',');
		float speedup = stof(token);		
		speedup_table[name]=speedup;	
	}

    for (map<string, float>::iterator it = speedup_table.begin(); it != speedup_table.end(); it++){
        sortedSpeedupVec.push_back(it->second);
    }
    if(reverse)
        sort(sortedSpeedupVec.begin(), sortedSpeedupVec.end(),cmpfunc_asc);
    else
        sort(sortedSpeedupVec.begin(), sortedSpeedupVec.end(),cmpfunc_desc);
	//setup device table (used in static_greedy)
    for(int j =0; j < sortedSpeedupVec.size(); j++){ // check task with the highest speedup
        float _key = sortedSpeedupVec[j];
        for (map<string,float>::iterator it = speedup_table.begin(); it != speedup_table.end(); it++){
            if(it->second == _key){
                    sortedSpeedupTasks.push_back(it->first);
#ifdef DEBUG
            		printf("name : %s and speedup : %f\n", it->first.c_str(), speedup_table[it->first]);
#endif
                    break;                    
            }
        }
				
    }
   	for (map<string, float>::iterator it = speedup_table.begin(); it != speedup_table.end(); it++){
        vector<float>::iterator fit;
        float _key = it->second;
        fit=find(sortedSpeedupVec.begin(), sortedSpeedupVec.end(), _key);
        int index = fit - sortedSpeedupVec.begin();
        if (index < nGPUs)
            device_table[it->first]=index;
        else 
              device_table[it->first]=nGPUs-1; // assign last device 
#ifdef DEBUG
        printf("name : %s and allocated device : %d\n", it->first.c_str(), device_table[it->first]);
#endif
    }

}

void GlobalScheduler::setupProcInfos(string dir_of_profile_file){

    //netname,batch size(ascending order), computil(%),dram usage(%)
    //
    ifstream infile(dir_of_profile_file);
    string line;
    string token;
    string key;
    float coreutil, memusage;
    while(getline(infile,line)){
        istringstream ss(line);
        getline(ss,token,',');
		string name = token;
		getline(ss,token,',');
		string batchsize = token;
        key = name+"-"+batchsize;
        getline(ss,token,',');
		coreutil = stof(token);
//        getline(ss,token,',');
		memusage = 0;
    ProcInfo *temp = (ProcInfo *)malloc(sizeof(ProcInfo));
    temp->compUtil = coreutil;
    temp->memUsage = memusage;
        Profile[key]=temp;
    }
   return;
}

// setups up the maximum batch for each network, for now we are going to use a fixed,static way
void GlobalScheduler::setupMaxBatchnDelay(string mb_file){
	ifstream infile(mb_file);
	string line;
	string token;
	while(getline(infile,line)){
		istringstream ss(line);
		int i=0;
		getline(ss,token,',');
		string name = token;		
		getline(ss,token,',');
		string device = token;
		tuple<string, string> temp;
		temp=make_tuple(name,device);
		getline(ss,token,',');
		mb_table[temp]=stoi(token);
        getline(ss,token,',');
        md_table[temp]=stoi(token);
#ifdef DEBUG
		printf("name : %s , device : %s , max batch size : %d, delay : %d \n", name.c_str(), device.c_str(), mb_table[temp], md_table[temp]);
#endif 
	}
   
}

void GlobalScheduler::setupOptCores(string optcore_file){
    ifstream infile(optcore_file);
	string line;
	string token;
	while(getline(infile,line)){
		istringstream ss(line);
		int i=0;
		getline(ss,token,',');
		string name = token;		
		getline(ss,token,','); // skip a token for now, every value is '1'
		getline(ss,token,',');
		optcore_table[name]=stof(token);
#ifdef DEBUG
		printf("name : %s and opt core value :  %f\n", name.c_str(), optcore_table[name]);
#endif 
	}

}

void GlobalScheduler::setupNetNames(string net_file)
{
    ifstream infile(net_file);
	string line;
	string token;
	while(getline(infile,line)){
		istringstream ss(line);
		int i=0;
		getline(ss,token,'.');
		string name = token;
		printf("name pushed in netnames : %s \n", name.c_str());
        net_names.push_back(name);
        agingTable[name]=0;
	}

}


// next function is used for testing and profiling
void GlobalScheduler::setupMaxBatch(int maxBatch){
	tuple<string, string> temp;
	for(int i=0; i<net_names.size();i++){
	string name = net_names[i];
	temp = make_tuple(name, "cpu");
	mb_table[temp]=maxBatch;
//	printf("name : %s and device : cpu batch size : %d\n", name.c_str(), mb_table[temp]);
	tuple<string, string> temp;
	temp = make_tuple(name, "gpu");
	mb_table[temp]=maxBatch;
//	printf("name : %s and device : gpu batch size : %d\n", name.c_str(), mb_table[temp]);
	}		
}

void GlobalScheduler::setMaxBatch(string name, string type, int max_batch)
{
    tuple<string, string> query = make_tuple(name, type);
    mb_table[query]=max_batch;
}

void GlobalScheduler::setupMaxDelay(int maxDelay){
	tuple<string, string> temp;
	for(int i=0; i<net_names.size();i++){
	string name = net_names[i];
	temp = make_tuple(name, "cpu");
	md_table[temp]=maxDelay;
//	printf("name : %s and device : cpu batch size : %d\n", name.c_str(), mb_table[temp]);
	tuple<string, string> temp;
	temp = make_tuple(name, "gpu");
	md_table[temp]=maxDelay;
//	printf("name : %s and device : gpu batch size : %d\n", name.c_str(), mb_table[temp]);
	}		

}



void GlobalScheduler::setNGPUs(int nGPU){
nGPUs=nGPU;
}

int GlobalScheduler::getNGPUs(){
    return nGPUs;
}

djinn::TensorDataOption GlobalScheduler::getDataOtption(string name){
    return input_dataoption_table[name];
}
int GlobalScheduler::getNumOfInputTensors(string name){
    return  input_datanum_table[name];
}


void GlobalScheduler::printProfileInfo(){
printf("total %d items profiled ! \n ", int(Profile.size()));

for (map<string , ProcInfo*>::iterator  mit = Profile.begin();  mit != Profile.end(); mit++){
	printf("%s , compUtil: %f, memUsage: %f \n", mit->first.c_str(), mit->second->compUtil, mit->second->memUsage);
	}
	printf("end of profiled infos ! \n");
}


int GlobalScheduler::getMaxBatch(string name ,string type){
    // the following should not fail if setup was called correctly
	map<tuple<string, string>, int>::iterator it;
	tuple<string, string> query = make_tuple(name, type);
	it=mb_table.find(query);
    if (it == mb_table.end()){
        printf("cant find max batch for %s \n", name.c_str());
        exit(1);
    }
	return it->second;
    
}

int getNextPowerof2(int n){ // if this function receives 2^n , return 2^(n+1)
    int ret_val =1;
    while (ret_val <=n){
        ret_val = ret_val *2;
    }
    return ret_val;
}


int GlobalScheduler::getMaxDelay(string reqname,string type, int QueueSize){
    // the following should not fail if setup was called correctly
	map<tuple<string, string>, int>::iterator it;
	tuple<string, string> query = make_tuple(reqname, type);
	it=md_table.find(query);
    if (it == md_table.end()){
        printf("cant find max delay for %s \n", reqname.c_str());
        exit(1);
    }
    int mb =  getMaxBatch(reqname, type);
    if (QueueSize >= mb) return 0; // if queue already has enough batch, just return

    if (!isBatchingAdaptive){
        return it->second;
    }
    // return adaptive delay 
    int delay=0;
    mb = getNextPowerof2(QueueSize);
    if (type=="gpu"){
        double upperbound = tableModel.findValueGPU(reqname, 1, 0);
        double T_mb = tableModel.findValueGPU(reqname, mb, 0);
        double lowerbound = T_mb/ mb ;
        double ArivInterval = (double)getTaskArivInterval(reqname);
        if (ArivInterval < lowerbound || ArivInterval > upperbound || (GPU_SLO - T_mb <0 )) {
#ifdef DEBUG 
        printf("[AB] lowerbound : %lf , upperbound : %lf for task %s\n",lowerbound,upperbound,reqname.c_str());
        printf("[AB] ArivInterval: %lf, no need to wait \n", ArivInterval);
#endif 
                return 0; //no need to wait
        }
        delay = int(((mb - QueueSize)* ArivInterval));
        if(delay > (GPU_SLO - T_mb)) delay =  int (GPU_SLO - T_mb);
#ifdef DEBUG 
        printf("[AB] lowerbound : %lf , upperbound : %lf for task %s\n",lowerbound,upperbound,reqname.c_str());
        printf("[AB] ArivInterval: %lf, delay : %d \n", ArivInterval, delay);
#endif 

    }
    else if(type=="cpu"){
        double upperbound = tableModel.findLatencyCPU(reqname, 1, 16);
        double T_mb = tableModel.findLatencyCPU(reqname, mb, 16)/ mb ;
        double lowerbound = T_mb / mb ;
        double ArivInterval = (double)getTaskArivInterval(reqname);
        if (ArivInterval < lowerbound || ArivInterval > upperbound || (GPU_SLO - T_mb <0 )){
#ifdef DEBUG 
        printf("[AB] lowerbound : %lf , upperbound : %lf for task %s for CPU \n",lowerbound,upperbound,reqname.c_str());
        printf("[AB] ArivInterval: %lf, no need to wait \n", ArivInterval);
#endif 

                return 0; //no need to wait
        }
         delay = int(((mb - QueueSize)* ArivInterval));
        if(delay > (GPU_SLO - T_mb)) delay =  int (GPU_SLO - T_mb);
#ifdef DEBUG 
        printf("[AB] lowerbound : %lf , upperbound : %lf for task %s\n",lowerbound,upperbound,reqname.c_str());
        printf("[AB] ArivInterval: %lf, delay : %d \n", ArivInterval, delay);
#endif 

    }

	return delay;
}


vector<shared_ptr<TaskSpec>> GlobalScheduler::staticGreedy(SysInfo *SysState){
    vector<shared_ptr<TaskSpec>> decision;
    for (map<string,queue<shared_ptr<request>>>::iterator it = SysState->ReqListHashTable.begin(); it != SysState->ReqListHashTable.end(); it++){
        if(it->second.size()){
            shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>();
            pTSpec->ReqName=it->first;
            pTSpec->BatchSize=-1;
            pTSpec->DeviceId =device_table[it->first];
            decision.push_back(pTSpec);
        }
        else{
#ifdef DEBUG
            printf("task %s has EMPTY QUEUE\n", it->first.c_str());
#endif 
        }
    }
	//device_table was already setted up in "setupSpeedup" function
	return decision;
}

int GlobalScheduler::getWeightedRRCnt(string ReqName, int batch){
        return int(weightTable.findValueGPU(ReqName,batch,0));

}

void GlobalScheduler::checkandRefreshCnt(SysInfo* SysState){
    if(refresh_cnt>=REFRESH_RATE){
        for(map<string, int>::iterator it = SysState->perTaskGenCnt.begin(); it != SysState->perTaskGenCnt.end(); it++){
            it->second=0;
        }
        refresh_cnt=0;

    }
}


vector<shared_ptr<TaskSpec>> GlobalScheduler::RRScheduler(SysInfo *SysState, bool weighted){
#ifdef DEBUG
    if(weighted) printf("wrr scheduler called \n");
#endif 
    vector<shared_ptr<TaskSpec>> decision;
    map<string, bool> needDecision;
    vector<bool> idleDevices;
    needDecision = getTasksNeedScheduling(SysState);
    int numNeedDecision=0;
    for(map<string ,bool>::iterator it = needDecision.begin(); it != needDecision.end(); it++){
        if(it->second) numNeedDecision++;
    }
    if (!numNeedDecision) return decision;
    // checkandRefreshCnt(SysState);
    //refresh_cnt++;
    // 2. get idle devices
    int numidleDevices=0;
#ifdef USE_CPU
    for(int i=0; i<nGPUs+nCPUs; i++){ //iterate through all devices

#else
    for(int i=0; i<nGPUs; i++){ //iterate through all GPUs
#endif 
		if((SysState-> perDeviceToExecList[i] + SysState->perDeviceBatchList[i]->size())<2){ 
            idleDevices.push_back(true);
            numidleDevices++;
        }
        else idleDevices.push_back(false);	
	}
   
    if(!numidleDevices) {
#ifdef DEBUG
        printf("No idle devices! \n");
#endif 
    return decision; // no idle devices
    }
        // 3. get lowest gen cnt among tasks and get lowest priority task
    string LowestTask;
    string theTask;
    int lowestgencnt;
    //int maxgencnt;
    lowestgencnt=2^31;
    for (int i=0; i<sortedSpeedupTasks.size(); i++){
        if(needDecision[sortedSpeedupTasks[i]]) {
            if( SysState->perTaskGenCnt[sortedSpeedupTasks[i]] < lowestgencnt){
                lowestgencnt=SysState->perTaskGenCnt[sortedSpeedupTasks[i]];
                theTask=sortedSpeedupTasks[i];
            }
            LowestTask=sortedSpeedupTasks[i];
        }
    }
    //4. update gencnt if needed
    for (int i=0; i<sortedSpeedupTasks.size(); i++){
        if(!needDecision[sortedSpeedupTasks[i]]) {
            if(SysState->perTaskGenCnt[sortedSpeedupTasks[i]] < lowestgencnt)
                SysState->perTaskGenCnt[sortedSpeedupTasks[i]]=lowestgencnt;
        }
#ifdef DEBUG
        printf("%s gen cnt : %d \n", sortedSpeedupTasks[i].c_str(), SysState->perTaskGenCnt[sortedSpeedupTasks[i]]);
#endif 
    }
    // 4. make decision 
    int MadeDecision=0;
    int index=0;
    while(MadeDecision< numidleDevices && MadeDecision < numNeedDecision){
    if (!idleDevices[index]){ index++;continue;}
    for (int i=0; i<sortedSpeedupTasks.size(); i++){ // ensure that the first task is compared
        if ( needDecision[sortedSpeedupTasks[i]]){
            theTask=sortedSpeedupTasks[i];
            lowestgencnt=SysState->perTaskGenCnt[theTask]; //lowest needs to be updated 
            break;
        }
    }
#ifdef DEBUG
    printf("%s chosen for comparing min gen cnt : %d \n", theTask.c_str(), SysState->perTaskGenCnt[theTask]);
#endif 

    //get task that has lowest gen count
    for (int i=0; i<sortedSpeedupTasks.size(); i++){
        if(needDecision[sortedSpeedupTasks[i]]){
#ifdef DEBUG
            printf("%s : %d V.S lowestgencnt: %d \n",sortedSpeedupTasks[i].c_str(), SysState->perTaskGenCnt[sortedSpeedupTasks[i]], lowestgencnt);
#endif 
            if( SysState->perTaskGenCnt[sortedSpeedupTasks[i]] < lowestgencnt){
                lowestgencnt=SysState->perTaskGenCnt[sortedSpeedupTasks[i]];
                theTask=sortedSpeedupTasks[i];
            }
        }
     }
#ifdef DEBUG
    printf("%s chosen for min gen cnt : %d \n", theTask.c_str(), SysState->perTaskGenCnt[theTask]);
#endif 

    shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>(); 
    pTSpec->DeviceId=index;
    pTSpec->BatchSize=-1;
    pTSpec-> ReqName=theTask;
    if (weighted){
        //get estimated batch size
        //
        tuple<string, string> temp;
	    temp = make_tuple(theTask, "gpu");
        int batch = min_int( mb_table[temp],SysState->ReqListHashTable[theTask].size());
        SysState->perTaskGenCnt[theTask] += getWeightedRRCnt(theTask,batch);
    }
    else
       SysState->perTaskGenCnt[theTask]++;

    decision.push_back(pTSpec);
    index++;
    MadeDecision++;

    }
	return decision;
}


vector<shared_ptr<TaskSpec>> GlobalScheduler::dynamicGreedy(SysInfo *SysState){
    vector<shared_ptr<TaskSpec>> decision;
    map<string, bool> needDecision;
    map<string, int> decisionCnt;
    vector<bool> idleDevices;
    needDecision = getTasksNeedScheduling(SysState);
    int numNeedDecision=0;
    for(map<string ,bool>::iterator it = needDecision.begin(); it != needDecision.end(); it++){
        if(it->second) numNeedDecision++;
        decisionCnt[it->first]=0;
    }
    if (!numNeedDecision) return decision;
    // 2. get idle devices
    int numidleDevices=nGPUs;
    /*for(int i=0; i<nGPUs; i++){ //iterate through all GPUs
		if(!SysState-> perDeviceBatchList[i]->size()){ 
            idleDevices.push_back(true);
            numidleDevices++;
        }
        else idleDevices.push_back(false);	
	}
    if(!numidleDevices) {
    return decision; // no idle devices
    }*/

    /*string LowestTask;
    for (int i=0; i<sortedSpeedupTasks.size(); i++){
        if(needDecision[sortedSpeedupTasks[i]]) {
            LowestTask=sortedSpeedupTasks[i];
        }
    }*/
    int MadeDecision=0;
    int index=0;
    int round=1;
    while(MadeDecision< numidleDevices){
        shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>(); 
        for (int i=0; i<sortedSpeedupTasks.size(); i++){
            if(needDecision[sortedSpeedupTasks[i]] && decisionCnt[sortedSpeedupTasks[i]] < round ){
#ifdef DEBUG 
                printf("%s will be schedled to device %d \n", sortedSpeedupTasks[i].c_str(), index);
#endif
                pTSpec->ReqName=sortedSpeedupTasks[i];
                pTSpec->DeviceId=index;
                pTSpec->BatchSize=-1;
                decisionCnt[sortedSpeedupTasks[i]]++;
                break;
            }
        }
        //if (!isGPU(index)) theTask = LowestTask;    
        decision.push_back(pTSpec);
        MadeDecision++;
        index++;
        if (MadeDecision % numNeedDecision ==0) // if every task was scheduled preprare next round 
                round++;
    }
	return decision;
}

vector<shared_ptr<TaskSpec>> GlobalScheduler::FIFOScheduler(SysInfo *SysState){
    vector<shared_ptr<TaskSpec>> decision;
    map<string, uint64_t> waitingTime;   
    vector<bool> idleDevices;
    // 1. get task's most waited request(type) : front task's { current time stamp - 'start' time stamp }
    for (map<string,queue<shared_ptr<request>>>::iterator it = SysState->ReqListHashTable.begin(); it != SysState->ReqListHashTable.end(); it++){
        ReqMtxVec[it->first]->lock();
        bool notEmpty = !it->second.empty();
        if(notEmpty) {
                if(!it->second.front()) waitingTime[it->first] = 0; // such case is possible, when you do NOT lock ReqListHashTable when accepting new requests, checks as soon as request is made
                else {
                    waitingTime[it->first]= (getCurNs() - it->second.front()->getStart());
                }
        }
        ReqMtxVec[it->first]->unlock();
        if(!notEmpty) waitingTime[it->first] = 0;
        
    }
    // 2. get idle devices
	for(int i=0; i<nGPUs; i++){ //iterate through all GPUs

		if((SysState->perDeviceToExecList[i]+SysState->perDeviceBatchList[i]->size())<2) idleDevices.push_back(true);
        else idleDevices.push_back(false);	
        //idleDevices.push_back(true);
	}

    // 3. sort waitingTime in descending order 
    vector<pair<string, uint64_t>> vec;
    for (map<string, uint64_t>::iterator it = waitingTime.begin(); it != waitingTime.end(); it++)
            vec.push_back(make_pair(it->first, it->second));
     sort(vec.begin(), vec.end(), cmpfunc_map_value_desc_uint64);
#ifdef DEBUG
    for(int i =0; i < vec.size(); i++){
        printf("[SCHEDULER] task : %s, waited : %lu ns \n", vec[i].first.c_str(), vec[i].second);
    }
#endif
    // 4. setup decision
    for (int i =0; i < vec.size(); i++){
        if (vec[i].second !=0){
//            for(int j=0; j<nGPUs+nCPUs; j++){
            for(int j=0; j<nGPUs; j++){

		        if(idleDevices[j]){
                    shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>();
                    pTSpec->DeviceId=j;
                    pTSpec->ReqName=vec[i].first;
                    pTSpec->BatchSize=-1;
                    idleDevices[j]=false;
                    decision.push_back(pTSpec);
                    break;
                }

	        }        
        }
        else break; // the last task is 0 if sorted descending
    }
	return decision;
}



double GlobalScheduler::getEstimatedWaitExecTime(SysInfo *SysState, string ReqName, int deviceID){
    double latency =0;
    //1. get estimated waiting latency for this device to wait given batch size
    if(isGPU(deviceID))
        latency += getMaxDelay(ReqName, "gpu",SysState->WaitingTable[ReqName]);
    else 
        latency += getMaxDelay(ReqName, "cpu",SysState->WaitingTable[ReqName]);
    return latency;
}


double GlobalScheduler::getEstimatedExecTime(SysInfo *SysState, string ReqName, int deviceID){
    double latency=0;
    if(isGPU(deviceID)) latency = tableModel.findValueGPU(ReqName, SysState->WaitingTable[ReqName],0);
    else latency = tableModel.findLatencyCPU(ReqName, SysState->WaitingTable[ReqName],16);
    return latency;
}

double GlobalScheduler::getEstimatedBatchWaitTime(SysInfo *SysState, string ReqName, int deviceID  ){
    double latency = 0;
    for(int i =0; i < SysState->perDeviceBatchList[deviceID]->size();i++){
        latency += getEstimatedExecTime(SysState,ReqName, deviceID);
        latency += getEstimatedWaitExecTime(SysState,ReqName, deviceID);
    }
    return latency;
}


float linearInterpolation(float x1, float y1, float x2, float y2, int p){   
    return (((y2-y1)/(x2-x1))*(p-x1))+y1;   
}

float GlobalScheduler::getGPUUtil(string name, int batch){
    string key1,key2; 
    float x1,x2,y1,y2;
    vector<int> batches = {1,2,4,8,16,32};
    int size=batches.size();
    for(int i=0; i<size-1 ; i++)
    {
       if( batches[i] <= batch && batch <batches[i+1] ){
           key1=name+"-"+to_string(batches[i]);
           key2=name+"-"+to_string(batches[i+1]);
        return linearInterpolation(batches[i],Profile[key1]->compUtil, batches[i+1],Profile[key2]->compUtil,batch);
       }
    }
    key1=name+"-"+to_string(batches[size-2]);
    key2=name+"-"+to_string(batches[size-1]);
    return linearInterpolation(batches[size-2],Profile[key1]->compUtil, batches[size-1],Profile[key2]->compUtil,batch);    
}

void GlobalScheduler::updateGPUUtil(SysInfo *SysState, int gpuid, string name, int batchsize, bool add){
    float util = getGPUUtil(name, batchsize);
    float org_util = SysState->GPUUtil[gpuid]; 
#ifdef DEBUG
    printf("Device %d before update : %lf \n",gpuid,org_util );
#endif 

    if(add) {
 //       SysState->GPUUtil[gpuid] = util+org_util <=100 ? util+org_util : 100;
       SysState->GPUUtil[gpuid] = util+org_util;

#ifdef DEBUG
    printf("Device %d add: %lf \n",gpuid,util);
#endif 

    }    
    else {
//        SysState->GPUUtil[gpuid] = org_util - util >=0 ? org_util-util : 0;
        SysState->GPUUtil[gpuid] = org_util - util;

#ifdef DEBUG
    printf("Device %d subtract: %lf \n",gpuid,util);
#endif 
    }
}
void GlobalScheduler::resetGPUUtil(SysInfo *SysState, int gpuid){
#ifdef DEBUG
    printf("reseting Device %d utilization %lf to 0 \n", gpuid, SysState->GPUUtil[gpuid]);
#endif
    SysState->GPUUtil[gpuid]=0;
}


vector<shared_ptr<TaskSpec>> GlobalScheduler::noScheduler(SysInfo *SysState){
    vector<shared_ptr<TaskSpec>> decision;
	for(map<string, queue<shared_ptr<request>>>::iterator it=SysState->ReqListHashTable.begin(); it != SysState->ReqListHashTable.end();it++ ){
      if(it->second.size() && !(SysState->perDeviceBatchList[0]->size() +  SysState->perDeviceToExecList[0])){
		shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>();
        pTSpec->DeviceId=0;
        pTSpec->ReqName=it->first;
        pTSpec->BatchSize=-1;
        decision.push_back(pTSpec);
      }   
    }
	return decision;    
}


void GlobalScheduler::setupMPSInfo(SysInfo *SysState, string capfile){
    //map<int, vector>::iterator it;
    ifstream infile(capfile);
    string line;
    string token;
    while(getline(infile,line)){
		istringstream ss(line);
		getline(ss,token,',');
		int devindex=stoi(token);
		getline(ss,token,',');
        int cap=stoi(token);
        proxy_info* pPInfo = (proxy_info*)malloc(sizeof(proxy_info));
        pPInfo->dev_id = devindex;
        pPInfo->cap = cap;
        vector<int>::iterator it; 
        it = find(SysState->perDevMPSCap[devindex].begin(), SysState->perDevMPSCap[devindex].end(),cap);
        if (it != SysState->perDevMPSCap[devindex].end()){
            pPInfo->dedup_num=1;

        }
        else{
            SysState->perDevMPSCap[devindex].push_back(cap);
            pPInfo->dedup_num=0;
        }
        pPInfo->isConnected=false;
        pPInfo->isSetup=false;
        SysState->perDevMPSInfo[devindex].push_back(pPInfo);

        printf("dev index : %d, mpscap: %d \n",devindex, cap );
	}
}

//gives all available to task with highest speedup
vector<shared_ptr<TaskSpec>> GlobalScheduler::gangScheduler(SysInfo *SysState){

    vector<shared_ptr<TaskSpec>> decision;
    map<string, bool> needDecision;
    needDecision= getTasksNeedScheduling(SysState);
    for(int i =0; i < sortedSpeedupTasks.size(); i++)
    {
       if(needDecision[ sortedSpeedupTasks[i]]){
            for(int j=0; j < nGPUs; j++){
                shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>();
                pTSpec->DeviceId=j;
                pTSpec->BatchSize=-1;
                pTSpec->ReqName= sortedSpeedupTasks[i];
                decision.push_back(pTSpec);
            }
            break;
       }

    }
    return decision;
}

//statically associates GPU (JUST FOR TESTING)
vector<shared_ptr<TaskSpec>> GlobalScheduler::staticMPSScheduler(SysInfo *SysState){
    vector<shared_ptr<TaskSpec>> decision;
    map<string, bool> needDecision;
    needDecision= getTasksNeedScheduling(SysState);
    // this scheudler is only going to accept two tasks, resnet18, squeezenet
    int threshold;
    for(int i =0; i < sortedSpeedupTasks.size(); i++){
        if ((sortedSpeedupTasks[i] == "vgg16" || sortedSpeedupTasks[i] == "dcgan") && needDecision[sortedSpeedupTasks[i]]){
            if (sortedSpeedupTasks[i] == "vgg16"){
                for(int j=0; j<nGPUs; j++){
                    proxy_info *pPInfo = SysState->perDevMPSInfo[j][0];
                    if (pPInfo -> cap == 100 || pPInfo -> cap == 50)
                            threshold=2;
                    else
                            threshold=1;
                    if (SysState->perProxyBatchList[{pPInfo->dev_id,pPInfo->cap}]->size() <=threshold){
                         shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>();
                         pTSpec->DeviceId=j;
                         pTSpec->BatchSize=-1;
                         pTSpec->ReqName= sortedSpeedupTasks[i];
                         pTSpec->CapSize = SysState->perDevMPSInfo[j][0]->cap;
                         decision.push_back(pTSpec);
                    }
                }
            }
            else if (sortedSpeedupTasks[i] == "dcgan") {
                    for(int j=0; j<nGPUs; j++){
                    proxy_info *pPInfo = SysState->perDevMPSInfo[j][1];
                    if (pPInfo -> cap == 100 || pPInfo -> cap == 50)
                            threshold=2;
                    else
                            threshold=1;

                    if (SysState->perProxyBatchList[{pPInfo->dev_id,pPInfo->cap}]->size() <=threshold){
                         shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>();
                         pTSpec->DeviceId=j;
                         pTSpec->BatchSize=-1;
                         pTSpec->ReqName= sortedSpeedupTasks[i];
                         pTSpec->CapSize = SysState->perDevMPSInfo[j][1]->cap;
                         decision.push_back(pTSpec);
                    }
                }

            }
                
        }
            
    }
    return decision;
}

// return cap id with given task spec, returns 0 if there is an error
int GlobalScheduler::getMPSCapID(shared_ptr<TaskSpec> pTask, SysInfo *SysState){
    int deviceID = pTask->DeviceId;
    int task_cap=pTask-> CapSize;
    int ret=-1;
    int prev_cap=1000;
    for (int i =0; i< SysState->perDevMPSCap[deviceID].size(); i++){
        //find best fit among remaining caps
        if(SysState->perDevMPSCap[deviceID][i] >= task_cap && (SysState->perDevMPSCap[deviceID][i] - task_cap) < (prev_cap - task_cap)){
            ret=i;
            prev_cap = SysState->perDevMPSCap[deviceID][i];
        }
    }
    return ret;
}

void GlobalScheduler::setupSLO(string slofile){
    ifstream infile(slofile);
	string line;
	string token;
	while(getline(infile,line)){
		istringstream ss(line);
		getline(ss,token,',');
		string name = token;		
		getline(ss,token,',');
		int slo = stoi(token);
        SLOTable[name]=slo;
#ifdef DEBUG
		printf("name : %s , slo: %d\n", name.c_str(), SLOTable[name]);
#endif 
	}

}
void GlobalScheduler::setupAvailvGPU(string specfile){
    ifstream infile(specfile);
	string line;
	string token;
	while(getline(infile,line)){
		istringstream ss(line);
		getline(ss,token,',');
		string name = token;		
		while(getline(ss,token,',')){
		    int cap = stoi(token);
            perTaskavailvGPU[name].push_back(cap);
        }
//#ifdef DEBUG
		printf("name : %s, caps: ", name.c_str());
        for(vector<int>::iterator it = perTaskavailvGPU[name].begin(); it != perTaskavailvGPU[name].end(); it++){
            printf("%d ",*it);
        }
        printf("\n");
//#endif 
	}

}

void GlobalScheduler::setupEstModels(){
    model.setup();
}


double getSendInputOverhead(int batch){
    // it took 5ms to send an input over IPC 
    return double(5)*(batch/32);
}

double GlobalScheduler::getEstLatency(string task, int batch, int cap){
    //input+sending overhead+base latency
    return getSendInputOverhead(batch)+model.get_baselatency(task,batch,cap);
}

int GlobalScheduler::getMaxEstLatency(string task, int batch){
    double max_lat=0;
    for(int i=0; i < perTaskavailvGPU[task].size(); i++){
        double est_lat = getEstLatency(task,batch ,perTaskavailvGPU[task][i]);
        if(est_lat > max_lat)
            max_lat=est_lat;
    }
    return int(max_lat)*1.15;
}

double GlobalScheduler::getInterferedLatency(string mytask, int mybatch, int mycap, string cotask, int cobatch, int cocap){
    return int(getSendInputOverhead(mybatch)+model.get_latency(mytask,mybatch, mycap, cotask,cobatch,cocap));
}

vector<shared_ptr<TaskSpec>> GlobalScheduler::SLOScheduler(SysInfo *SysState){
    assert(USE_MPS=true); // this scheduler will only be used when MPS is turned on
    vector<shared_ptr<TaskSpec>> decision;
    map<string, int> waitingTime;
    map<string, int> batchEstimate;
    map<string, int> budget;
   //get batch estimation for each TaskSpec
    batchEstimate = getEstimatedBatchSize(SysState);

    // 1. get each task's budget
    for (map<string,queue<shared_ptr<request>>>::iterator it = SysState->ReqListHashTable.begin(); it != SysState->ReqListHashTable.end(); it++){
        ReqMtxVec[it->first]->lock();
        bool notEmpty = !it->second.empty();
        if(notEmpty) {
                if(!it->second.front()) waitingTime[it->first] = 0; // such case is possible, when you do NOT lock ReqListHashTable when accepting new requests, checks as soon as request is Made
                else {
                    waitingTime[it->first]=(getCurNs() - it->second.front()->getStart())/1000000;
                    budget[it->first]=  SLOTable[it->first] - waitingTime[it->first] - getMaxEstLatency(it->first,batchEstimate[it->first]);
#ifdef DEBUG
                    printf("task:%s ,slo:%d ,waitingTime: %d, batch: %d,maxEstLat: %d\n",it->first.c_str(), SLOTable[it->first], waitingTime[it->first],batchEstimate[it->first],getMaxEstLatency(it->first,batchEstimate[it->first]) );
#endif
                }
        }
        ReqMtxVec[it->first]->unlock();
        if(!notEmpty) waitingTime[it->first] =0;
    }

    // no need to schedule
    if (budget.size() == 0)
        return decision;

#ifdef DEBUG
    for(map<string, int>::iterator it = budget.begin(); it != budget.end(); it++){
        printf("[SCHEDULER] task : %s, budget : %d ms \n", it->first.c_str(), it->second);
    }
#endif

    // 2. sort budget in ascending order
    vector<pair<string, int>> vec;
    for (map<string, int>::iterator it = budget.begin(); it != budget.end(); it++)
            vec.push_back(make_pair(it->first, it->second));
     sort(vec.begin(), vec.end(), cmpfunc_map_value_asc);

     // 3. get # of tasks of each proxy 
     map<pair<int,int>, int> perProxyTasks;
     map<pair<int,int>, int> perProxyScheduledTasks;
    for(int i=0; i<nGPUs; i++){ //iterate through all GPUs
        for(int j=0; j<SysState->perDevMPSInfo[i].size(); j++){
            proxy_info* pPInfo = SysState->perDevMPSInfo[i][j];
            int nTasks = SysState->perProxyBatchList[{pPInfo->dev_id,pPInfo->cap}]->size() \
                         + SysState->perProxyExecutingList[{pPInfo->dev_id,pPInfo->cap}];
            perProxyTasks[{pPInfo->dev_id,pPInfo->cap}]=nTasks;
            perProxyScheduledTasks[{pPInfo->dev_id,pPInfo->cap}]=0;

        }
    }
    
    //make decisions   
    const int THRESHOLD=2;
    for (int i =0; i < vec.size(); i++){
        if(batchEstimate[vec[i].first]==0) continue;  //this happens because waiting time is meaasured after batch estimation
        bool scheduled=false;
        int dev_id=-1;
        int cap=-1;
        double min_lat=10000000;
       // get device which yields minimum latency 
        for(map<pair<int,int>, int>::iterator it=perProxyTasks.begin(); it!=perProxyTasks.end();it++){
            //skip busy proxys
            //get dedup num of proxy queue 
            //for now the only case a dedup happens is either 50 or 100
             int threshold;
            if (it->first.second == 50 || it->first.second == 100)
                threshold = THRESHOLD *2;
            else
                threshold = THRESHOLD;
            if(it->second + perProxyScheduledTasks[{it->first.first,it->first.second}]>=threshold) continue;
            for(int j =0; j<perTaskavailvGPU[vec[i].first].size(); j++){ //check whether proxy is available for GPU
                if (perTaskavailvGPU[vec[i].first][j] == it->first.second){
                    //check for corunning tasks
                    double estlatency;
                    bool isInterference=false;
                    int checkcap=100-(it->first.second);
                    string cotask;
                    int cobatch;
                    perProxyExecMtx[{it->first.first,checkcap}]->lock();
                    if (SysState->perProxyExecutingTask[{it->first.first,checkcap}]->size() != 0){
                        shared_ptr<TaskSpec> pTSpec = SysState->perProxyExecutingTask[{it->first.first,checkcap}]->at(0);
                         isInterference=true;
                         cotask=pTSpec->ReqName;
                         cobatch= pTSpec->BatchSize;
                         pTSpec.reset();
                    }
                    perProxyExecMtx[{it->first.first,checkcap}]->unlock();
                    if (isInterference && checkcap!=0){
#ifdef DEBUG
                        printf("found interference!\n");
#endif 
                        estlatency=getInterferedLatency(vec[i].first, batchEstimate[vec[i].first], it->first.second\
                                            ,cotask,cobatch,checkcap);

                    }
                   else
                        estlatency=getEstLatency(vec[i].first,batchEstimate[vec[i].first],it->first.second);
                    if(estlatency<min_lat){
                        scheduled=true;
                        min_lat=estlatency;
                        dev_id=it->first.first;
                        cap=it->first.second;
                    }
                    break; // there is only one match in available GPU list anyway
                }

            }
            //if(scheduled) break;
        }
        if(scheduled){
#ifdef DEBUG
            printf("task %s scheduled to proxy [%d,%d]\n", vec[i].first.c_str(), dev_id,cap);
#endif
            shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>();
            pTSpec->DeviceId=dev_id;
            pTSpec->ReqName=vec[i].first;
            //pTSpec->BatchSize=batchEstimate[vec[i].first];
            pTSpec->BatchSize=batchEstimate[vec[i].first];
            pTSpec->CapSize=cap;
            decision.push_back(pTSpec);
            perProxyScheduledTasks[{dev_id,cap}]+=THRESHOLD; // make sure a scheduled device does not get another task for this epoch 
        }
        
    }
	return decision;

}

void GlobalScheduler::checkandRefreshCredit(){
    float threshold = 0.5;
    //GPU
    //check whether half of the tasks exceed threshold(30%) of credit
    int cnt = net_names.size() / 2;
    for (map<string, int>::iterator it = gpu_credit_table.begin(); it != gpu_credit_table.end(); it++){
        if (it -> second < priority_table[it->first] * GPU_TAU * threshold) 
            cnt --;        
    }
    //if so refresth 
    if(cnt <= 0){
#ifdef DEBUG
        printf("[SCHEDULER] refreshing GPU credit \n");
#endif

        for (map<string, int>::iterator it = gpu_credit_table.begin(); it != gpu_credit_table.end(); it++){
            it -> second = priority_table[it->first] * GPU_TAU ;
        }           
    }
#ifdef DEBUG
    for (map<string, int>::iterator it = gpu_credit_table.begin(); it != gpu_credit_table.end(); it++){
            printf(" gpu credit of %s : %d \n",it->first.c_str(), it->second);
        }
#endif 

    //CPU
    //check whether half of the tasks exceed threshold(30%) of credit

    cnt = net_names.size() / 2;
    for (map<string, int>::iterator it = cpu_credit_table.begin(); it != cpu_credit_table.end(); it++){
        if (it -> second < 1 * CPU_TAU * threshold) 
            cnt --;        
    }
    //if so refresth 
    if(cnt <= 0){
#ifdef DEBUG
        printf("[SCHEDULER] refreshing CPU credit \n");
#endif


        for (map<string, int>::iterator it = cpu_credit_table.begin(); it != cpu_credit_table.end(); it++){
            it -> second = 1 * CPU_TAU ;
        }           
    }
#ifdef DEBUG
    for (map<string, int>::iterator it = cpu_credit_table.begin(); it != cpu_credit_table.end(); it++){
            printf("cpu credit of %s : %d \n",it->first.c_str(), it->second);
        }
#endif 
    return;
}

vector<shared_ptr<TaskSpec>> GlobalScheduler::creditScheduler(SysInfo *SysState){
    // init decision table 
    vector<string> decision;
    vector<shared_ptr<TaskSpec>> decision2;
    for (int i=0; i< net_names.size(); i++){
        decision.push_back("no");
    }   
    //get idle devices
    vector<bool> idleDevices;
    int numidleDevices=0;
    for(int i =0; i<nGPUs+nCPUs; i++){
        if(SysState->perDeviceBatchList[i]->size()) 
            idleDevices.push_back(false);
        else{
            idleDevices.push_back(true);
            numidleDevices++;
        }
    }
    if(numidleDevices == 0 ) {// such case is possible when previous scheduling has already taken care of all idle devices
#ifdef DEBUG
        printf("[SCHEDULER] exiting scheduler, no idle devices \n");
#endif
        return decision2;
    }
    // get tasks that need decision
    vector<string> needDecision;
    map<string, bool> isSetted;
    map<string, int> expectedBatch;
    int decision_cnt=0; // used for exiting later on 
    for(map<string,int>::iterator it=SysState->WaitingTable.begin(); it != SysState->WaitingTable.end();it++ ){
        if(it->second){ 
                needDecision.push_back(it->first);
                isSetted[it->first]=false;
                decision_cnt++;
                expectedBatch[it->first]=it->second;
        }
    }
    if (needDecision.size()==0)
        return decision2;
 
   
#ifdef DEBUG 
    printf("[SCHEDULER] size of need decision : %lu  and idle devices : %d \n", needDecision.size(),numidleDevices);
    for (int i =0; i < needDecision.size(); i++ )
        printf("%s,",needDecision[i].c_str());
    printf("\n");
#endif 

    //1) check if credit needs to be refreshed 
   checkandRefreshCredit();
    //2) assign tasks based on credit 
    for (int i = 0; i < nGPUs+nCPUs; i++){
        if (idleDevices[i]){
            //2-1) for idle GPU chose one with the most gpu credit
           if (i < nGPUs){
               int highest_credit = -100000; 
               string highest_task;
               for(map<string, int>::iterator it = gpu_credit_table.begin(); it != gpu_credit_table.end(); it ++){
                   if (highest_credit < it -> second && find(needDecision.begin(),needDecision.end(),it -> first) != needDecision.end() && !isSetted[it->first]){
#ifdef DEBUG
                       printf("found highest : %d \n", it -> second);
#endif 

                       highest_credit = it-> second;
                       highest_task = it->first;
                   }                    
               }
               decision[i]=highest_task; 
               idleDevices[i]=false;
               float delta = float(min_int(32, expectedBatch[highest_task])) / 32 ; 
#ifdef DEBUG
               printf("delta : %lf \n " , delta);
#endif 
               gpu_credit_table[highest_task] -= int(gpu_weightedEpochDelta[highest_task] * delta);

               isSetted[highest_task]=true;
               decision_cnt--;
           }        
        //2-2) for idle CPU, chose one with the highest cpu credit 
           else{
            int highest_credit = -100000; 
               string highest_task;
               for(map<string, int>::iterator it = cpu_credit_table.begin(); it != cpu_credit_table.end(); it ++){
                   if (highest_credit < it -> second && find(needDecision.begin(),needDecision.end(),it -> first) != needDecision.end() && !isSetted[it->first]){
                       highest_credit = it-> second;
                       highest_task = it->first;
                   }                    
               }
               decision[i]=highest_task; 
               idleDevices[i]=false;
                float delta = float(min_int(32, expectedBatch[highest_task])) / 32 ; 

               cpu_credit_table[highest_task] -= int(cpu_weightedEpochDelta[highest_task] * delta); 
               isSetted[highest_task]=true; 
               decision_cnt--;
            }  
        }   
        if (decision_cnt == 0) break; 

   }
   return decision2;
}

vector<shared_ptr<TaskSpec>> GlobalScheduler::vGPUwRRScheduler(SysInfo *SysState, bool weighted){
#ifdef DEBUG
        printf("sGPUwRRScheduler called \n");
#endif 
    vector<shared_ptr<TaskSpec>> decision;
    map<string, bool> needDecision;
    map<string, int> estimatedBatch;
    vector<bool> idleDevices;
    needDecision = getTasksNeedScheduling(SysState);

    int numNeedDecision=0;
    for(map<string ,bool>::iterator it = needDecision.begin(); it != needDecision.end(); it++){
        if(it->second) numNeedDecision++;
    }
    if (!numNeedDecision) return decision;

        // 1. get lowest gen cnt among tasks and get lowest speedup
    string LowestTask;
    string theTask;
    int lowestgencnt;
    //int maxgencnt;
    lowestgencnt=2^31;
    for (int i=0; i<sortedSpeedupTasks.size(); i++){
        if(needDecision[sortedSpeedupTasks[i]]) {
            if( SysState->perTaskGenCnt[sortedSpeedupTasks[i]] < lowestgencnt){
                lowestgencnt=SysState->perTaskGenCnt[sortedSpeedupTasks[i]];
                theTask=sortedSpeedupTasks[i];
            }
            LowestTask=sortedSpeedupTasks[i];
        }
    }
    //4. get lowest speedup task and update gencnt if needed
    for (int i=0; i<sortedSpeedupTasks.size(); i++){
        if(!needDecision[sortedSpeedupTasks[i]]) {
            if(SysState->perTaskGenCnt[sortedSpeedupTasks[i]] < lowestgencnt)
                SysState->perTaskGenCnt[sortedSpeedupTasks[i]]=lowestgencnt;
        }
    }
    estimatedBatch=getEstimatedBatchSize(SysState);

    // 2. make decision 
    int MadeDecision=0;
    vector<float> *tempUtil=new vector<float>();
    vector<int> *tempDecision=new vector<int>(); 
    for (int i =0; i<nGPUs;i++) {
            tempDecision->push_back(0);
            tempUtil->push_back(0);
            if((SysState->perDeviceBatchList[i]->size() + SysState->perDeviceToExecList[i]) ==0)
                resetGPUUtil(SysState,i);

    }

    while(MadeDecision < numNeedDecision){
    //update      
    for (int i=0; i<sortedSpeedupTasks.size(); i++){ 
        if ( needDecision[sortedSpeedupTasks[i]]){
            theTask=sortedSpeedupTasks[i];
            lowestgencnt=SysState->perTaskGenCnt[theTask]; //lowest needs to be updated 
            break;
        }
    }

    //get task that has lowest gen count
    for (int i=0; i<sortedSpeedupTasks.size(); i++){
        if(needDecision[sortedSpeedupTasks[i]]){
            if( SysState->perTaskGenCnt[sortedSpeedupTasks[i]] < lowestgencnt){
                lowestgencnt=SysState->perTaskGenCnt[sortedSpeedupTasks[i]];
                theTask=sortedSpeedupTasks[i];
            }
        }
     }
    bool isAvailable=false;
    float util = getGPUUtil(theTask, estimatedBatch[theTask]);
    float diff = -1000;
    int index=-1;
    for(int i=0;i<nGPUs; i++){ // search each GPU portion , best fit , but changed to worst fit

#ifdef DEBUG
        printf("device %d, batch+exec+sched list size : %lu \n", i, SysState->perDeviceBatchList[i]->size() + SysState->perDeviceToExecList[i] + tempDecision->at(i));
#endif 
        if(SysState->perDeviceBatchList[i]->size() + SysState->perDeviceToExecList[i] + tempDecision->at(i)> VGPU_QUEUE_CAP)  continue;
#ifdef DEBUG
        printf("device %d, real util: %lf \n", i, SysState->GPUUtil[i]);
       printf("device %d, calculated util: %lf \n", i, SysState->GPUUtil[i] + util + tempUtil->at(i));
#endif 

        if (SysState->GPUUtil[i] + util + tempUtil->at(i) > vGPU_THRESHOLD){
            continue;
        }

        if (vGPU_THRESHOLD - (SysState->GPUUtil[i] + util + tempUtil->at(i)) > diff){
            diff = vGPU_THRESHOLD - (SysState->GPUUtil[i] + util + tempUtil->at(i));
            index=i;
            isAvailable=true;
        }
    }
    if(! isAvailable) {//means there are currently no available device at the moment
        delete tempUtil;
        delete tempDecision;
        return decision;     
    }
    shared_ptr<TaskSpec> pTSpec = make_shared<TaskSpec>(); 
    pTSpec->DeviceId=index;
    pTSpec->BatchSize=estimatedBatch[theTask];
    pTSpec->ReqName=theTask;
    if (weighted){
        SysState->perTaskGenCnt[theTask] += getWeightedRRCnt(theTask,estimatedBatch[theTask]);
    }
    else
       SysState->perTaskGenCnt[theTask]++;
    decision.push_back(pTSpec);
    tempUtil->at(index)+=util;
    tempDecision->at(index)+=1;
    MadeDecision++;
    }
    delete tempUtil;
    delete tempDecision;
	return decision;

}
