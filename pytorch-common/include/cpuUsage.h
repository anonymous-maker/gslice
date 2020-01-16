#ifndef CPUUSAGE_H__
#define CPUUSAGE_H__
#include <vector>
#include <string>
#include <pthread.h>
#include <condition_variable>

namespace Utilization {

// enum for deciding which mode to measure utilization
// AGG(default) : Aggretated mode, sumup every thread's utilization
// IND : Individual mode, measure and prints every thread's utilization
enum MEASURE_MODE {AGG, IND};



class cpuUsage{
    public:
        cpuUsage();
        void setPID(int pid);
        void switchtoInd(); // switch mode agg -> imd
        void switchtoAgg(); // switch mode ind -> agg
        void setIncludeChild(bool flag);
        void startMeasure();
        void printnstopMeasure(); 
        void exitMeasureThread();
        void setInterval(float interval); // setsup interval
        std::vector<unsigned int> getTimes(std::string strpid);
        void measureUtilization();
        void measureAGGUtilization();
        void measureINDUtilization();
    private:
        MEASURE_MODE _mode;
        pthread_t _monThread; // internal thread
        int _pid;
        float _interval; // checking interval
        bool _exitflag; // intialized on construction , setted to 'true' by exitMeasureThread,  
       // bool _startflag; // intialized on construction, setted to 'true' by startMeasure()
      //  bool printflag; // initialized on consturuction, setted to 'true' by printnstopMeasure()
        bool _includechild; // initilized on construction, setted with setIncludeChild 
          std::string strpid; 
     //   std::mutex pauseMtx;
      //  std::condition_variable pauseCV; 

};

//warpper function for calling pthread for an object 
void *cpuUsageWrapper(void *object);

void initMeasureThread(cpuUsage *mon, int pid, float interval); // start measure

}

#endif
