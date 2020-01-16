/*
 *  Copyright (c) 2015, University of Michigan.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

/**
 * @author: Johann Hauswald, Yiping Kang
 * @contact: jahausw@umich.edu, ypkang@umich.edu
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <map>
#include <glog/logging.h>
#include <time.h>
#include <nvml.h>
#include <queue>
#include <vector>

#include "scheduler.h"
#include "boost/program_options.hpp"
#include "socket.h"
#include "thread.h"
#include "tonic.h"
#include "ts_list.h"
#include "request.h"
#include "batched_request.h"
#include "common_utils.h"
#include "mon.h"
#include "input.h"
#include "ts_queue.h"

#define TOTAL_CORES 20
using namespace std;
namespace po = boost::program_options;
extern SysInfo ServerState; 
extern vector<mutex*> ReqCmpMtx;
extern vector<shared_ptr<condition_variable>> ReqCmpCV;
int TOTAL_CMPQUEUE=1;
const int TOTAL_SENDTHREAD=30;

GlobalScheduler gsc;
bool USE_GPU;
bool USE_MPS;
bool vGPU=false;
bool EXP_DIST=false;
FILE* pLogFile;
int CPUID;
bool EXIT;
//file + dir that containts required files
string COMMON;
string WEIGHT_DIR;
string NET_LIST;




po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")(
      "common,c", po::value<string>()->default_value("../common/"),
      "Directory with configs and weights")(
      "portno,p", po::value<int>()->default_value(8080),
      "Port to open DjiNN on")
      ("nets,n", po::value<string>()->default_value("nets.txt"),
       "File with list of network configs (.prototxt/line)")
("weights,w", po::value<string>()->default_value("weights/"),"Directory containing weights (in common)")
("gpu,g", po::value<bool>()->default_value(false), "Use GPU?")
("threadcnt,t",po::value<int>()->default_value(-1),"Number of threads to spawn before exiting the server. (-1 loop forever)")
("ngpu,ng", po::value<int>()->default_value(4),"number of gpus in system to run service")
("scheduler,s",po::value<string>()->default_value("no"),"scheduling mode : no, greedy, dedicated, proposed")
("profile,prof",po::value<string>()->default_value("profile.csv"),"the file which stores profile value of each network" )
("max_delay, md",po::value<int>()->default_value(0),"maximum amount of time for waiting(ms)")
("ncpu,nc",po::value<int>()->default_value(1),"number of virtual cpu devices to be spawned" )
("mps,m",po::value<bool>()->default_value(false),"flag setting whether mps is being used")
("adaptive_batch,ab",po::value<bool>()->default_value(false),"flag setting whether adaptive batch is used")
("local,l", po::value<bool>()->default_value(true), "flag setting whether experiment will runned locally" )
("input_txt, it", po::value<string>()->default_value("server_input.txt"), "txt file which holds img input files" )
("req_gen_txt, rqt", po::value<string>()->default_value("req_gen_spec.txt"), "txt file which holds req gen specs" )
("exp_dist,ed",po::value<bool>()->default_value(false),"flag setting whether exp distrubtion will be used");



  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return vm;
}

int main(int argc, char* argv[]) {
  // Main thread for the server
  // Spawn a new thread for each request
  uint64_t start, end;
  po::variables_map vm = parse_opts(argc, argv);

  USE_GPU = vm["gpu"].as<bool>();
  USE_MPS = vm["mps"].as<bool>();

  int numofGPU=vm["ngpu"].as<int>();
  int numofCPU=vm["ncpu"].as<int>();
  if (numofCPU != 0 && TOTAL_CORES % numofCPU){
    printf("exiting server, cannot equally devide cores among %d CPUs\n", numofCPU);
    exit(1);
  }
  
  // set the CPU ID to be number of GPUs
  CPUID=numofGPU;

  // global variable needed to load all models to gpus at init
  COMMON = vm["common"].as<string>();
  NET_LIST = vm["nets"].as<string>();
  WEIGHT_DIR = vm["weights"].as<string>();

 
  //initiate threads and scheduler/monitor
 // init scheduler
  bool isAdaptiveBatch = vm["adaptive_batch"].as<bool>();
  EXP_DIST = vm["exp_dist"].as<bool>();
  string mode=vm["scheduler"].as<string>();
  if(!gsc.setSchedulingMode(mode, isAdaptiveBatch)){
    printf("exiting server due to undefined scheduling mode\n");
    exit(1);
  }
  if (isAdaptiveBatch) gsc.setupTableModel("gpuprof.csv", "cpuprof.csv");

// setup some global & systemwide variables 

  gsc.setNGPUs(numofGPU);
  gsc.setupNumCPUDevices(numofCPU);
#ifdef DEBUG
  printf("numofGPU : %d , numofCPU : %d \n", numofGPU, numofCPU);
#endif 
  gsc.setMaxCPUCores(TOTAL_CORES);
  //gsc.setupMonitor();
  gsc.setupNetNames(NET_LIST);
  gsc.setupInputSpecs("input_tensor_specs.txt");

  //string profile_dir= vm["profile"].as<string>();
  //gsc.setProcInfos(profile_dir);
  
   //int maxDelay=vm["max_delay"].as<int>();

   gsc.setupMaxBatchnDelay(string("MaxBatch.txt")); // for now lets just hard code the file name 
  
  //TEMP : setup speedup per benchmark, for speeup experiment
  if (mode=="static"){
    gsc.setupPriority(string("latency.txt"), false);
  }
  else if (mode=="dynamic" || mode=="rr"){
    gsc.setupPriority(string("latency.txt"), true);
  }
    else if(mode == "mps"){
    gsc.setupPriority("priority.txt",false);
    //gsc.setupWeightedEpoch("weightedepoch.txt");
  }
    else if(mode=="gang") gsc.setupPriority(string("latency.txt"), false);
  else if(mode=="mps_test") gsc.setupPriority(string("latency.txt"), false);
  else if(mode =="slo"){
          gsc.setupSLO(string("SLO.txt"));
          gsc.setupAvailvGPU(string("AvailvGPU.txt"));
          gsc.setupEstModels();
  }
  else if(mode == "wrr"){
          gsc.setupWeights("weight.csv");
          gsc.setupPriority(string("latency.txt"), true);
  }
  else if(mode == "wrr_sgpu"){
          gsc.setupWeights("weight.csv");
          gsc.setupPriority(string("latency.txt"), true);
          gsc.setupProcInfos("util.csv");
         gsc.printProfileInfo();
         vGPU=true;
  }

   
  pLogFile=fopen("log.txt","w");
  fprintf(pLogFile,"timestamp,task,TaskID,deviceID,RECV_TO_REQ,REQ_TO_BATCH,BATCH_TO_EXEC,EXEC,TO CMPQ,CMPQ_WRITE\n");
  fflush(pLogFile);

  pthread_t tid;

  //setup input data in memory
  bool local = vm["local"].as<bool>();
  vector<ReqGenStruct> ReqGens;
  if(local){
    string input_txt = vm["input_txt"].as<string>();
    readImgData(input_txt.c_str(), IMG_POOL);
  }
  ReqGens = gsc.setupReqGenSpecs(vm["req_gen_txt"].as<string>());
    
  if(USE_MPS) gsc.setupMPSInfo(&ServerState, "ThreadCap.txt");
  tid=initServerThread(numofGPU);
    
  if(!USE_MPS)
    sleep((gsc.getNGPUs() + gsc.getNCPUs()) * 8) ;
  else{
    sleep((gsc.getNGPUs() + gsc.getNCPUs()) *3) ;
  }

 // how many threads to spawn before exiting
  // -1 to stay indefinitely open
   int thread_cnt = 0;
  int total_thread_cnt = vm["threadcnt"].as<int>();
  //initClearThread();


// init threads that need to run in background
  pthread_t perfmont = initPerfMonThread(&ServerState);
  //initGPUUtilThread();
//  initEpochThread();

  if(local){
vector<string> strNames;
 for(int i =0; i < ReqGens.size(); i++){
         string* str = new string(ReqGens[i].StrReqName);
        strNames.push_back(*str);
 }
 for(int i =0; i < TOTAL_CMPQUEUE; i++){
        //ReqCmpMtx.push_back(new mutex());
        //ReqCmpCV.push_back(make_shared<condition_variable>());
        for (int j=0; j<TOTAL_SENDTHREAD; j++){
            initSendResultsThread(i);
        }


 }
    //generate request filling threads 
    vector<pthread_t> tids;

    printf("started to generate requests \n");

    for(int i =0; i < ReqGens.size(); i++){
             tids.push_back(initFillQueueThread(&ReqGens[i]));
    }
    int doExit=0;
    EXIT =false ;//global bool used as a flag to exit periodic threads
    while(1){
        sleep(1);
        doExit=0;
        for(int i =0; i < ReqGens.size(); i++){
            string str(ReqGens[i].StrReqName);
            if (ServerState.CompletedTable[str] == ReqGens[i].nrequests) doExit++;
        }
        if(doExit ==  ReqGens.size()) break;
    }
    printf("all requests have been computed ! \n");
    fflush(pLogFile);
    EXIT=true;
    //close all sockets if using MPS
    if(USE_MPS){
        for (int i=0; i<gsc.getNGPUs();i++) {
            for (int j =0; j<ServerState.perDevMPSInfo[i].size(); j++){
                SOCKET_close(ServerState.perDevMPSInfo[i][j]->in_fd,false);
                SOCKET_close(ServerState.perDevMPSInfo[i][j]->out_fd,false);
            }
        }
    }
    //wait for threads
   for(int i =0; i < tids.size(); i++){
            if (pthread_join(tids[i], NULL) != 0) {
                LOG(FATAL) << "Failed to join.\n";
              }
    } 

    if (pthread_join(perfmont, NULL) != 0) {
                LOG(FATAL) << "Failed to join.\n";
    }
  }
  else{
 // Listen on socket
            int socketfd = SERVER_init(vm["portno"].as<int>());

            listen(socketfd, 20);
          
            LOG(ERROR) << "Server is listening for requests on " << vm["portno"].as<int>()<<endl;
          while (1) {
         
            pthread_t new_thread_id;
              int client_sock = accept(socketfd, (sockaddr*)0, (unsigned int*)0);
#ifdef DEBUG
              cout << "accepting new socket , "<<client_sock << endl; 
#endif 
          
            if (client_sock == -1) {
              LOG(ERROR) << "Failed to accept.\n";
              continue;
            }    
           
             new_thread_id = initRequestThread(client_sock);
            ++thread_cnt;
            if (thread_cnt == total_thread_cnt) {
              if (pthread_join(new_thread_id, NULL) != 0) {
                LOG(FATAL) << "Failed to join.\n";
              }
              break;
            }
          }
  }
  return 0;
}

