
#include <fstream>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include <glog/logging.h>
#include <boost/chrono/thread_clock.hpp>
#include <time.h>
#include <cstring>
#include <assert.h>
#include <condition_variable>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>   
#include <mutex>  

#include "gpu_proxy.h" 
#include "batched_request.h"
#include "socket.h"
#include "common_utils.h"
#include "scheduler.h"

using namespace std;
//PROXY RELATED FUNCS and VARS



//need to find a bettery way to store map, lets change this to reading file/store in the future.
unordered_map<string, int> mapping={{"vgg16", 0}, {"resnet18", 1}, {"alexnet", 2}, {"squeezenet", 3}, {"dcgan-gpu", 4}};

int connectGPUProxyOut(int gpuid, int threadcap, int dedup){
    struct sockaddr_un d;
    int d_fd;
    float* in;   
    memset(&d, 0, sizeof(struct sockaddr_un));
    d_fd=socket(AF_UNIX, SOCK_STREAM, 0); 
    if (d_fd == -1) {
        printf("SOCKET ERROR = %s\n", strerror(errno));
        exit(1);
    }
    d.sun_family=AF_UNIX;
    char buffer[50];
    char idbuffer[20];
    strcpy(buffer,"/tmp/gpusock_output_");
    snprintf(idbuffer,sizeof(idbuffer),"%d_%d_%d",gpuid,threadcap, dedup);
    strcat(buffer,idbuffer);
//#ifdef DEBUG
    printf("[PROXY] connecting to %s \n", buffer);
//#endif
    strcpy(d.sun_path, buffer);
    int r2=connect(d_fd, (struct sockaddr*)&d, sizeof(d));
    if (r2 == -1) {
        printf("SOCKET ERROR = %s\n", strerror(errno));
        exit(1);
    }
//#ifdef DEBUG
      printf("[PROXY] Connected to proxy server of gpu%d\n",gpuid);
//#endif
    return d_fd;

}

int connectGPUProxyIn(int gpuid, int threadcap, int dedup){
struct sockaddr_un d;
    int d_fd;
    float* in;   
    memset(&d, 0, sizeof(struct sockaddr_un));
    d_fd=socket(AF_UNIX, SOCK_STREAM, 0); 
    if (d_fd == -1) {
        printf("SOCKET ERROR = %s\n", strerror(errno));
        exit(1);
    }
    d.sun_family=AF_UNIX;
    char buffer[50];
    char idbuffer[20];
    strcpy(buffer,"/tmp/gpusock_input_");
    snprintf(idbuffer,sizeof(idbuffer),"%d_%d_%d",gpuid,threadcap,dedup);
    strcat(buffer,idbuffer);
//#ifdef DEBUG
    printf("[PROXY] connecting to %s \n", buffer);
//#endif
    strcpy(d.sun_path, buffer);
    int r2=connect(d_fd, (struct sockaddr*)&d, sizeof(d));
    if (r2 == -1) {
        printf("SOCKET ERROR = %s\n", strerror(errno));
        exit(1);
    }
//#ifdef DEBUG
      printf("[PROXY] Connected to proxy server of gpu%d\n",gpuid);
//#endif
    return d_fd;

    }

static __inline__ unsigned long long rdtsc(void)
{
        unsigned long long int x;
        __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
        return x;
}

int sendRequest(int in_fd, char net_name[16], int rid, torch::Tensor input_tensor){
    int sock_elts=1;
    int len=4;

    int ret=write(in_fd, (void *)&len,sizeof(int));
    if(ret<=0){
        perror("write");
    }
         for(int i=0; i<input_tensor.sizes().size(); i++){
        len = input_tensor.size(i);
        ret=write(in_fd, (void *)&len,sizeof(int));
        sock_elts*=len;
    }
    ret=write(in_fd, (void *)&sock_elts,sizeof(int));
    float *raw_data=input_tensor.data<float>();  
     int r;
#ifdef DEBUG
    printf("[PROXY] sending name %s,rid: %d ,sock_elts: %d \n", net_name, rid ,sock_elts);
//    printf("----------- tsc: %llu\n", rdtsc());
#endif
r=SOCKET_send(in_fd, (char*)raw_data, sock_elts*sizeof(float), false);

   //ret=write(in_fd, (char *)in,sock_elts*sizeof(float));

     //send batch ID 
    r=write(in_fd, (void*)&rid,sizeof(int));
    int jobID=-1;
    // do a table lookup and send corresponding job ID
    for (unordered_map<string,int>::iterator it= mapping.begin(); it != mapping.end(); it++){
        if(!strcmp(net_name, it->first.c_str())){
            jobID=it->second;
            break;
        }
    }
    if(jobID == -1) {
        printf("no such request as %s", net_name);
        exit(0);
    }
    r=write(in_fd, (void*)&jobID,sizeof(int));
#ifdef DEBUG
    //printf("[PROXY] completed sending\n");
#endif 
    if (r == -1){
         printf("SEND ERROR - RID: %d\n", rid);
         printf("SOCKET ERROR = %s\n", strerror(errno));
    }
        return r;
}

//returns reqID, and stores output data in pointer
int recvResult(int out_fd, float **output, int* output_len){
    int rid=0;   
    int ret;
#ifdef DEBUG
    //printf("waiting for output \n");
#endif 
    if ((ret=read(out_fd, &rid, sizeof(int)))>0){
#ifdef DEBUG
    //printf("successfuly received rid : %d \n",rid);
#endif 
    }

    //rid=header[0];
    //size=header[1];
    //*output=(float*)malloc(size);
    //*output_len=size/(sizeof(float));
    //memset(*output, 0, size);
    //int cur_read=0;
    //while(cur_read!=size){
    //    cur_read+=read(out_fd, (void*)(*output)+cur_read, size-cur_read);
    //}
    return rid;
}
