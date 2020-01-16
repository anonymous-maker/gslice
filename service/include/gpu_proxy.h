#ifndef _PROXY_H__
#define _PROXY_H__
#include "torchutils.h"
#include <torch/script.h>
#include "tonic.h"
#include <mutex> 
typedef struct {
    int dev_id;
    int cap;
    int dedup_num; // especially needed if there are more than one pair of {dev_id, cap}
    int in_fd;
    int out_fd;
    bool isConnected;//indecating whether thread is conneced to proxy 
    bool isSetup; // indicating whether thread is all set up, CV, mtx, batch list
    std::mutex *sendMtx;
} proxy_info;

typedef struct {
    char net_name[16];
    int rid;
    int sock_elts; // total # of float
} netinfo_packet;

typedef struct {
    netinfo_packet* p;
    float* data;
    unsigned int in_size;
    unsigned int out_size;
}dataqueue_elem;

typedef struct {
    int rid;
    float *data; 
    int output_size;
}queue_elem;

int connectGPUProxyIn(int gpuid, int thredcap, int dedup);
int connectGPUProxyOut(int gpuid, int threadcap, int dedup);
int sendRequest(int in_fd,  char net_name[16], int rid, torch::Tensor input_tensor); // return : send size 
int recvResult(int out_fd, float** output,  int* output_size); // return : reqID , output -> output data , out_size -> # of floats 
int closeAllSockets();
#else
#endif
