#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <unistd.h>
#include <vector>
#include <string>
#include <pthread.h>
#include <sys/types.h>
#include <dirent.h>
#include <time.h>

#include "cpuUsage.h"

namespace Utilization{


bool _startflag;
//std::mutex startMtx;
//std::condition_variable startCV;

bool _printflag;
std::mutex pauseMtx;
std::condition_variable pauseCV;

static uint64_t getCurNs(){
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        uint64_t t = ts.tv_sec * 1000 * 1000 * 1000 + ts.tv_nsec;
        return t;
}

cpuUsage::cpuUsage(){
    _exitflag=false;
    _printflag=false;
    _includechild=true;
    _mode=AGG;
}

void cpuUsage::switchtoInd(){
    if(_mode==AGG){
    exitMeasureThread();
    //reset flags
    _exitflag=false;
    _printflag=false;
    // change mod
    _mode=IND;
    // measure against the same pid with allocated interval
    initMeasureThread(this, _pid, _interval);
    }
}

// not completed!
void cpuUsage::switchtoAgg(){
 if(_mode==IND){
    exitMeasureThread();
    //reset flags
    _exitflag=false;
    _printflag=false;
    // change mod
    _mode=AGG;
    // measure against the same pid with allocated interval
    initMeasureThread(this, _pid, _interval);
    }
}

void cpuUsage::startMeasure(){
  //  {
 //   std::lock_guard<std::mutex> lk(startMtx);
    _startflag=true;
 //   }
 //   startCV.notify_one();
}

void cpuUsage::setPID(int pid){
    _pid=pid;
}

void cpuUsage::setIncludeChild(bool flag){
    _includechild=flag;
}

void cpuUsage::exitMeasureThread(){
    _exitflag=true;
    pthread_join(_monThread, NULL);
}

void cpuUsage::setInterval(float interval){
    _interval=interval;
}

void cpuUsage::printnstopMeasure(){
    {
    std::lock_guard<std::mutex> lk(pauseMtx);
    _printflag=true;
    }
    pauseCV.notify_one();
}
std::vector< unsigned int> cpuUsage::getTimes(std::string strpid) {
    std::ifstream proc_stat;
    if (_mode==AGG){
      //  std::cout << strpid << std::endl;
        proc_stat.open("/proc/"+strpid+"/stat",  std::ios::in);
    }
    else if (_mode == IND){
         std::string strtgid = std::to_string(_pid);
      //  std::cout << strtgid << ","<<strpid<<std::endl;
        proc_stat.open("/proc/"+strtgid+"/task/"+strpid+"/stat",  std::ios::in);
    }
    std::vector<std::string> strTokens;
    for (std::string time; proc_stat >> time; )strTokens.push_back(time);
    std::vector< unsigned int> times;
    for (int i = 13; i<=16; i++){
            times.push_back(stoul(strTokens[i])); // utime, stime, cutime, cstime 
    }
     times.push_back(stoul(strTokens[19])); //numthread
     times.push_back(stoul(strTokens[21])); //starttime
    return times;
}
float getUpTime(){
    std::ifstream proc_uptime("/proc/uptime");                       
    std::vector<std::string> strTokens;
    for (std::string time; proc_uptime >> time; )strTokens.push_back(time);  
    return std::stof(strTokens[0]);
}


void  cpuUsage::measureAGGUtilization() {
   _monThread=pthread_self();

    std::string  strpid = std::to_string(_pid);
    float tick_time = sysconf(_SC_CLK_TCK);
    while(!_exitflag){
    //wait for the starting signal
  //  std::cout << "waiting for start signal "<< std::endl;
    while(!_startflag){
            usleep(_interval*1000000); // sleep for some to avoid excessive cpu hogging, accepts microseconds
            if (_exitflag ) return;
    }
    _startflag = false;
//    std::cout << "start measure!"<<std::endl;
    std::vector< unsigned int> cpu_times = getTimes(strpid);
    unsigned int base_total_time; 
    if (_includechild)
        base_total_time = cpu_times[0]+cpu_times[1] + cpu_times[2] + cpu_times[3];
    else
        base_total_time = cpu_times[0] + cpu_times[1];
    float uptime = getUpTime();
        //wait for the signal
    /* while(!printflag){
            usleep(_interval * 1000000);
           if (_exitflag )break;
    }   
    _printflag=false;*/
    uint64_t start=getCurNs();
    std::unique_lock<std::mutex> lk(pauseMtx);
    pauseCV.wait(lk,[]{return _printflag;});
    
    cpu_times = getTimes(strpid);
    unsigned int total_time; 
    if (_includechild)
        total_time = cpu_times[0]+cpu_times[1] + cpu_times[2] + cpu_times[3];
    else
        total_time = cpu_times[0] + cpu_times[1];
    uint64_t end = getCurNs();
    float exec_seconds = float(end-start ) / 1000000000;
    float cpu_usage = 100 * ((  (total_time-base_total_time)/ tick_time)/ (exec_seconds));
    std::cout<<"thread_cnt: "<<cpu_times[4]-1<<" usage: "<<cpu_usage<<std::endl;// -1 for  THIS thread(checking cpu usage) 
    _printflag=false; 
    lk.unlock();
    }   
    
}
std::vector<std::string> getTIDs(std::string strtgid){
    std::vector<std::string> dirs;
    std::string root_dir = "/proc/"+strtgid+"/task";
    DIR* dirp = opendir(root_dir.c_str());
    struct dirent *dp;
    while((dp = readdir(dirp)) != NULL ) {
        dirs.push_back(dp->d_name);
       // std::cout<<dp->d_name<<std::endl;
    }
    return dirs;
}
void cpuUsage::measureINDUtilization(){
    _monThread=pthread_self();

    std::string  strtgid = std::to_string(_pid);

//    std::vector<std::string> tids=getTIDs(strtgid);
//    tids.erase(tids.begin(), tids.begin()+2);
//    float tick_time = sysconf(_SC_CLK_TCK);
//    std::vector<unsigned int> base_total_time_vec;
//    for (int i=0; i<tids.size(); i++) {
//        base_total_time_vec.push_back(0);
//     }
    while(!_exitflag){
        //wait for the starting signal
        //  std::cout << "waiting for start signal "<< std::endl;
        while(!_startflag){
            usleep(_interval*1000000); // sleep for some to avoid excessive cpu hogging, accepts microseconds
            if (_exitflag )return;
        }
      _startflag = false;

        std::vector<std::string> tids=getTIDs(strtgid);
        tids.erase(tids.begin(), tids.begin()+2);
        float tick_time = sysconf(_SC_CLK_TCK);
        std::vector<unsigned int> base_total_time_vec;
        for (int i=0; i<tids.size(); i++) {
            base_total_time_vec.push_back(0);
        }
          //    std::cout << "start measure!"<<std::endl;
            for(int i=0;i<tids.size(); i++){
            std::vector< unsigned int> cpu_times = getTimes(tids[i]); 
            if (_includechild)
                base_total_time_vec[i] = cpu_times[0]+cpu_times[1] + cpu_times[2] + cpu_times[3];
            else
                base_total_time_vec[i] = cpu_times[0] + cpu_times[1];
         }
        //wait for the si
    
        uint64_t start=getCurNs();
/*        while(!_printflag){
            usleep(_interval * 1000000);
            if (_exitflag )break;
        }   
        _printflag=false;*/
          std::unique_lock<std::mutex> lk(pauseMtx);
   pauseCV.wait(lk,[]{return _printflag;});

        unsigned int total_time; 

  
        std::cout<< "thread_cnt :  " <<tids.size() << std::endl;
        for(int i=0;i<tids.size(); i++){
            std::vector< unsigned int> cpu_times = getTimes(tids[i]);
            unsigned int total_time; 
            if (_includechild)
                total_time = cpu_times[0]+cpu_times[1] + cpu_times[2] + cpu_times[3];
            else
                total_time = cpu_times[0] + cpu_times[1];
        uint64_t end = getCurNs();
        float exec_seconds=float(getCurNs()-start)/1000000000;
        float cpu_usage = 100 * ((  (total_time-base_total_time_vec[i])/ tick_time)/ (exec_seconds));
            std::cout<<"threadID:  "<<tids[i]<<" usage: "<<cpu_usage<<" duration: "<<exec_seconds<<std::endl;
        }
     _printflag = false;
     lk.unlock();

    } 
}

void cpuUsage::measureUtilization(){
    switch(_mode){
        case AGG:
            measureAGGUtilization();
        case IND:
            measureINDUtilization();
    }
}

void *cpuUsageWrapper(void *object){
    ((cpuUsage*)object) -> measureUtilization();
}
void initMeasureThread(cpuUsage *mon, int pid, float interval){
    pthread_attr_t attr;                                                                             
    pthread_attr_init(&attr);  
    mon->setPID(pid);
    mon->setInterval(interval);
    pthread_t thread;
    pthread_attr_setstacksize(&attr, 8*1024 * 1024); // set memory size, may need to adjust in the future, for now set it to 1MB                                    
    if (pthread_create(&thread, &attr, cpuUsageWrapper, mon) != 0){                                                            
        std::cout << "Failed to create a request handler thread.\n";  
        return;

    }
} 

} // namespace Utilization
