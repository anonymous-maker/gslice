#include <stdio.h>
#include <stdlib.h>
#include <nvml.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include<signal.h>
#include<unistd.h>
#include <vector>
#include <algorithm>


#include "mon.h"

using namespace std;
//global flag for checking to end monitor program
int finish;


//following used to be local variables
float maxBandwidth; // this needs to change for heterogenous system
struct timespec stop, start;
double seconds;
vector<int> gpuids;
nvmlDevice_t* temp;

char *timeStamp(){                                                                                                                                                    
      char *pTimeStamp = (char *)malloc(sizeof(char) * 16);                                                                                                           
      time_t ltime;                                                                                                                                                   
      ltime=time(NULL);                                                                                                                                               
      struct tm *tm;                                                                                                                                                  
      struct timeval tv;                                                                                                                                              
      int millis;                                                                                                                                                     
      tm=localtime(&ltime);                                                                                                                                           
      gettimeofday(&tv,NULL);                                                                                                                                         
      millis =(tv.tv_usec) / 1000 ;                                                                                                                                   
      sprintf(pTimeStamp,"%02d:%02d:%02d:%03d", tm->tm_hour, tm->tm_min, tm->tm_sec,millis);                                                                          
      return pTimeStamp;                                                                                                                                              
  }                             

int initNVML()
{  
   // need to check wheter operated properly 
    //initialize NVML library
   nvmlReturn_t result; 
   result = nvmlInit();
   if (result != NVML_SUCCESS){
   printf("Init unsuccessful! error code : %d\n", result);
   	return 0;
   }

   return 1;
}


int initMonitor(int gpuid, nvmlDevice_t *gpu0)
{
     
    nvmlReturn_t result; 
    int num = gpuid;
    nvmlMemory_t memInfo;
    if ( !( -1 <num || num < 4 )){
	printf("gpu ID must be between 0 ~ 3 ");
	return 0;
    }	
    if (find(gpuids.begin(),gpuids.end(),num) != gpuids.end()){
    // found  
      printf("gpuid: %d already initialized! \n",num);
      return 1; 
    }
   // not found
   printf(" initializing gpuid %d monitor!\n",num);
   gpuids.push_back(num);
    
   result=nvmlDeviceGetHandleByIndex(num, gpu0);
   if (result != NVML_SUCCESS){
	printf("Device Init Unsuccessful! error code : %d\n", result);
	return 0;
    }
    unsigned int g0PcieBandwidth, currLinkGen;
    nvmlDeviceGetCurrPcieLinkWidth(*gpu0, &g0PcieBandwidth);
    nvmlDeviceGetCurrPcieLinkGeneration(*gpu0, &currLinkGen); 
    nvmlDeviceGetMemoryInfo(*gpu0,&memInfo);

    if (currLinkGen == 1)
	    maxBandwidth = 0.25 * g0PcieBandwidth;
    else if(currLinkGen == 2)
	    maxBandwidth = 0.5 * g0PcieBandwidth;
    else if(currLinkGen == 3)
	    maxBandwidth = 0.985 * g0PcieBandwidth;
    else //gen4
	    maxBandwidth = 1.969 * g0PcieBandwidth;
    maxBandwidth = 15.76; // decided to hardcode this value
//    printf("link width : %u \n",g0PcieBandwidth);
//    printf("link generation : %u\n",currLinkGen);
//    printf("link max bandwidth : %f GB/s \n",maxBandwidth);
//    printf("Total memory : %3f MB\n",(float)memInfo.total/1024/1024);
    return 1;

}

void getUtilization(nvmlDevice_t gpu0, float *util_info)
{
	unsigned int tx, rx;
	int memUsage;
	unsigned int  pciInfo;
	bool found=false;
	nvmlMemory_t memInfo;
	nvmlUtilization_t utilInfo;   
	nvmlReturn_t result;
	result = nvmlDeviceGetMemoryInfo(gpu0,&memInfo);
	if (result != NVML_SUCCESS){
	printf("Reading memory info unsuccessful! error code : %d\n", result);
	return;
	}
	nvmlDeviceGetPcieThroughput(gpu0, NVML_PCIE_UTIL_TX_BYTES,&pciInfo);
	tx=pciInfo;
	nvmlDeviceGetPcieThroughput(gpu0, NVML_PCIE_UTIL_RX_BYTES,&pciInfo);
	rx=pciInfo;	
	tx = ((float)tx)/1024/1024;
	rx = ((float)rx)/1024/1024;
	tx = tx/maxBandwidth*100;
	rx = rx/maxBandwidth*100;       
	memUsage = ((float)memInfo.used) / memInfo.total*100;
	nvmlDeviceGetUtilizationRates(gpu0, &utilInfo);
	if (result != NVML_SUCCESS){
	printf("Reading utilization info unsuccessful! error code : %d\n", result);
	return;
	}

	util_info[0]=memUsage;
	util_info[1]=tx;
	util_info[2]=rx;
	util_info[3]=float(utilInfo.gpu);
//	printf("%s, %3f,%3f,%3f,%u,%u \n",time_stamp(),memUsage,tx,rx,utilInfo.gpu,utilInfo.memory);
	return;
 
}

int endNVML()
{
   nvmlReturn_t result; 
  // need to check wheter operated properly
   result = nvmlShutdown();
     if (result != NVML_SUCCESS){
    	printf("Ending NVML Unsuccessful! error code : %d\n", result);
   	return 0;
    }
   return 1;
}
