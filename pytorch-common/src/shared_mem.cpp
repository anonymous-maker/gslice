#include "shared_mem.h"

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

/*
| STATE | size | jobid |(output,used when needed, otherwise just data)|data ~ ~| 

*/
#define CON_OFFSET 0 
#define SIZE_OFFSET 4
#define JOB_OFFSET 8
#define DATA_OFFSET 12
#define OUTPUT_OFFSET 16 
#define TOTAL_SIZE 10485760
enum STATE{WAIT=0,INPUT_READY,COMPUTE,OUTPUT_READY,END};

//cretes(or open) shmem segment, returns shmem id, called by server
int SHMEM_create(const char *pathname, int proj_id){
	key_t key = ftok(pathname,proj_id);
    if (key==-1){
        perror("error in making key");
    }
	assert(key != -1);
	int shmid = shmget(key,TOTAL_SIZE,0666|IPC_CREAT);
	assert(shmid != -1);
	return shmid;
}

// attach shmem to provided pointers, 
int SHEMEM_register(int shmem_id, void* pData){
 	pData = (void *) shmat(shmem_id,(void*)0,0);
    return 0;
}

int SHMEM_putInput(int jobId, int size ,void* pData, float* input){
    //clear Data
    memset((char*)pData+DATA_OFFSET,0,TOTAL_SIZE - DATA_OFFSET);
    int *ip=(int*)((char*)pData+JOB_OFFSET);
    *ip=jobId;
    ip=(int *)((char*)pData+SIZE_OFFSET);
    *ip=size;
    memcpy((char*)pData+DATA_OFFSET,input,size);
    ip=(int*)pData;
    *ip=INPUT_READY;
	return 0;
}

int SHMEM_getInput(void* pData, float* input, int *jobid){
    int *pCon = (int *)pData;
    while( *pCon != INPUT_READY){usleep(1000);} 
    int *size = (int *)((char*)pData+SIZE_OFFSET);
    memcpy(input,(float*)((char*)pData+DATA_OFFSET),*size);
    *jobid = *(int *)((char*)pData+JOB_OFFSET);
    *pCon=COMPUTE;
	return 0;
}
int SHMEM_putOutput(void* pData, int *output){
    //clear and copy output to pData
    memset((char*)pData+OUTPUT_OFFSET,0,sizeof(int));
    memcpy((char*)pData+OUTPUT_OFFSET,output,sizeof(int));
    int *pCon = (int *)pData;
    *pCon=OUTPUT_READY;
	return 0;
}


int SHMEM_getOutput(int shmem_id, void *pData, int *ret){
    int *pCon = (int *)pData;
    while(*pCon != OUTPUT_READY ) {usleep(1000);}
    *pCon=WAIT;    
	return 0;
}

// called by frontend
int SHMEM_deregister(void* pData){
    int *pCon = (int *)pData;
    *pCon = END;
	shmdt(pData);
	return 0;
}

// called by proxys (make sure all has called deregister before calling destroy)
int SHMEM_destroy(int shmemid){
    shmctl(shmemid,IPC_RMID,NULL);
	return 0;
}

