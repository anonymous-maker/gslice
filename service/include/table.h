#ifndef _TABLE_H__
#define _TABLE_H__

#include <string.h>
#include <unordered_map>

using namespace std;
class PerfTable{
    public:
        PerfTable();
        ~PerfTable();
        double findLatencyCPU(string bench, int batch, int thread);
        double findValueGPU(string bench, int batch, int interference);
        void createTableCPU(string filename);
        void createTableGPU(string filename);
        void printTableContents(); // used for debugging
// privately used methods and variables
    private:
        bool checkWorkload(string s);
        unordered_map<string, double> CPUTable;
        unordered_map<string, double> GPUTable;
        vector<string> benchType;
        vector<int> threadNum;
        vector<int> batchSizeCPU;
        vector<int> batchSizeGPU;
        vector<int> interferenceAmount;
};
#endif 
