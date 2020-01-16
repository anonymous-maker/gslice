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
#ifndef _THREAD_H_
#define _THREAD_H_
#include "state.h"
#include "scheduler.h"

#include <pthread.h>
#include <stdio.h>
#include <vector>
#include <deque>
#include <mutex>

//#include "caffe/caffe.hpp"
#include "socket.h"
#include "tonic.h"
#include "request.h"
#include "batched_request.h"
#include "gpu_proxy.h"

#define NUM_CORE 20 // the number of total logical cores within the system

struct ArgStruct {
  int datalen;
  int GPUID;
  char  ReqName[MAX_REQ_SIZE];
  float *inData;
  
};




pthread_t initRequestThread(int sock);
//pthered_t request_med_thread_init(int sock, int gpu_id);
void* handleRequest(void* sock);

pthread_t initExecutionThread(batched_request *req);
void* handleExecution(shared_ptr<batched_request> input_info,string strReqName,shared_ptr<TaskSpec> pTask);

pthread_t initDeviceThread(int gpu_id);
void* initDevice(void *args);

pthread_t initProxyThread(proxy_info* pPInfo);
void* initProxy(void *args);

pthread_t initServerThread(int numGPU);
void* initServer(void* numGPU);

pthread_t initClearThread();
void* handlerClear(void *vp);

pthread_t initPerfMonThread(SysInfo *Serverstate);
void* PerfMon(void *vp);

pthread_t initGPUUtilThread();
void* GPUUtilMon(void *vp);

void warmupDevice(int devID);

pthread_t initEpochThread();
void* initEpoch(void *vp);

pthread_t initSendResultsThread(int id);
void* initSend(void *args);

pthread_t initFillQueueThread(ReqGenStruct *args);
void* initFill(void *args);

void sendBatchedResults(shared_ptr<batched_request> brp, string reqname); // defined in device_thread.cpp 

#else
#endif // #define _THREAD_H_
