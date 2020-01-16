#include "common_utils.h"
#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <stdint.h>
#include <ctime>
#include <sstream>


void printTimeStampWithName(const char* name, const char* premsg, FILE* fp){
    char buffer[30];
    struct timeval tv; 
    time_t curtime;
    gettimeofday(&tv, NULL); 
    curtime=tv.tv_sec;
    strftime(buffer,30,"%m-%d-%Y  %T.",localtime(&curtime));
    fprintf(fp,"[%s] %s%06ld : %s\n",name, buffer,tv.tv_usec,premsg);
}

void printTimeStamp(const char* premsg){
    char buffer[30];
    struct timeval tv; 
    time_t curtime;
    gettimeofday(&tv, NULL); 
    curtime=tv.tv_sec;
    strftime(buffer,30,"%m-%d-%Y  %T.",localtime(&curtime));
    printf("%s%06ld : %s\n",buffer,tv.tv_usec,premsg);
}
void printTimeStampWithName(const char* name, const char* premsg){
    char buffer[30];
    struct timeval tv; 
    time_t curtime;
    gettimeofday(&tv, NULL); 
    curtime=tv.tv_sec;
    strftime(buffer,30,"%m-%d-%Y  %T.",localtime(&curtime));
    printf("[%s] %s%06ld : %s\n",name, buffer,tv.tv_usec,premsg);
}
uint64_t getCurNs() {
   struct timespec ts; 
   clock_gettime(CLOCK_REALTIME, &ts);
   uint64_t t = ts.tv_sec*1000*1000*1000 + ts.tv_nsec;
   return t;
}

std::string uint64_to_string( uint64_t value ) {
	std::ostringstream os;
	os << value;
	return os.str();
}

