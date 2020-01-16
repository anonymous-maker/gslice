#ifndef COM_UTILS_H__
#define COM_UTILS_H__

#include <stdint.h>
#include <string>
void printTimeStamp(const char* premsg);
void printTimeStampWithName(const char* name, const char* premsg);
void printTimeStampWithName(const char* name, const char* premsg, FILE* fp);
uint64_t getCurNs();
std::string uint64_to_string(uint64_t value);
#else
#endif 
