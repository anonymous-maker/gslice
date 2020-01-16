
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

//PROXY RELATED FUNCS and VARS

int connectGPUProxyOut(int gpuid){
    struct sockaddr_un d;
    int d_fd;
    char test_packet[sizeof(netinfo_packet)];
    float* in;
    memset(&d, 0, sizeof(struct sockaddr_un));
    d_fd=socket(AF_UNIX, SOCK_STREAM, 0); 
    if (d_fd == -1) {
        printf("SOCKET ERROR = %s\n", strerror(errno));
        exit(1);
    }
    d.sun_family=AF_UNIX;
    char buffer[50];
    char idbuffer[11];
    strcpy(buffer,"/tmp/gpusock_output_");
    snprintf(idbuffer,sizeof(idbuffer),"%d",gpuid);
    strcat(buffer,idbuffer);
#ifdef DEBUG
    printf("[PROXY] connecting to %s \n ", buffer);
#endif
    strcpy(d.sun_path, buffer);
    int r2=connect(d_fd, (struct sockaddr*)&d, sizeof(d));
    if (r2 == -1) {
        printf("SOCKET ERROR = %s\n", strerror(errno));
        exit(1);
    }
#ifdef DEBUG
      printf("[PROXY] Connected to proxy server of gpu%d\n",gpuid);
#endif
    return d_fd;
}

int connectGPUProxyIn(int gpuid){
    struct sockaddr_un d;
    int d_fd;
    char test_packet[sizeof(netinfo_packet)];
    float* in;

    memset(&d, 0, sizeof(struct sockaddr_un));
    d_fd=socket(AF_UNIX, SOCK_STREAM, 0);
    if (d_fd == -1) {
        printf("SOCKET ERROR = %s\n", strerror(errno));
        exit(1);
    }
    d.sun_family=AF_UNIX;
    char buffer[50];
    char idbuffer[11];
    strcpy(buffer,"/tmp/gpusock_input_");
    snprintf(idbuffer,sizeof(idbuffer),"%d",gpuid);
    strcat(buffer,idbuffer);
#ifdef DEBUG
    printf("[PROXY] connecting to %s \n ", buffer);
#endif
    strcpy(d.sun_path, buffer);
    int r2=connect(d_fd, (struct sockaddr*)&d, sizeof(d));
    if (r2 == -1) {
        printf("SOCKET ERROR = %s\n", strerror(errno));
        exit(1);
    }
#ifdef DEBUG
      printf("[PROXY] Connected to proxy server of gpu%d\n",gpuid);
#endif
    return d_fd;
}

int sendRequest(int in_fd, char net_name[16], int rid, int sock_elts, float *data){
    netinfo_packet *sending = (netinfo_packet *) malloc(sizeof(netinfo_packet));
    strcpy(sending->net_name, net_name);
    sending->rid = rid;
    sending->sock_elts=sock_elts;

    float *in = (float*)malloc(sizeof(netinfo_packet)+sock_elts * sizeof(float));

    // concat data right behind the header 
    memcpy(in, sending,sizeof(netinfo_packet));
    memcpy(in+sizeof(netinfo_packet), data, sock_elts * sizeof(float));
    int r;
    r=send(in_fd, in, sizeof(netinfo_packet)+sock_elts*sizeof(float), 0);
    if (r == -1){
         printf("SEND ERROR - RID: %d\n", rid);
    }
    return r;
}

//returns reqID, and stores output data in pointer
int recvResult(int out_fd, float **output, int* output_len){
    int header[2];
    int rid;
    int size;
    void* data;
    read(out_fd, header, sizeof(int)*2);
    rid=header[0];
    size=header[1];
    *output=(float*)malloc(size);
    *output_len=size/(sizeof(float));
    memset(*output, 0, size);
    int cur_read=0;
    while(cur_read!=size){
        cur_read+=read(out_fd, (void*)(*output)+cur_read, size-cur_read);
    }
    return rid;
}
