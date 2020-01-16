#ifndef OPENCVUTILS_H // To make sure you don't declare the function more than once by including the header multiple times.
#define OPENCVUTILS_H

#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <pthread.h>
#include <queue>
#include <condition_variable>

#include "torchutils.h"
typedef struct _img_input{
    int n_height;
    int n_width;
    cv::Mat *pMat;
    std::queue<torch::Tensor> *pOutput_queue;
    std::condition_variable *poutCV;
    std::mutex *pLock; // shared threads processing same job , lock used when accesing output vector
    int MAX_THREAD; // maximum number of threads for this job
    int jobID;
    bool skip_resize;
} img_input; 

typedef struct _ThreadPool{
    std::vector<pthread_t> tids;
    std::queue<img_input> taskQueue;
    std::mutex inputQueueMtx;
    std::condition_variable cv;
    std::map<int,int> perRequstThreadCnt;
    std::mutex threadCntMtx; 
    int nextJobID; // used for allocating jobIDs
} ThreadPool;


//int getNextJobID(); //returns the global Pools next job ID 

cv::Mat preprocess(cv::Mat &image, int new_height, int new_width);

unsigned char* serializeMat(cv::Mat input_mat);
cv::Mat unserializeMat(unsigned char* serial_data);

void insertTaskinQueue(img_input *input);
void insertTaskinQueue(cv::Mat &input, int imagenet_row, int imagenet_col, int ncores, int jobID, 
                        std::queue<torch::Tensor> *computeQueue,std::mutex *compQueueMtx, 
                        std::condition_variable *compQueueCV, bool skip_resize);



//torch::Tensor parallelPreprocess(std::vector<cv::Mat> &input, int new_height, int new_width,int nthreads);
torch::Tensor serialPreprocess(std::vector<cv::Mat> input, int imagenet_row, int imagenet_col);

/*void preprocessAsync(
                std::queue<std::vector<cv::Mat>> *requestQueue,
               std::mutex reqQueueMtx *reqQueueMtx,
               std::condition_variable *reqQueueCV,
               std::queue<torch::Tensor> *computeQueue,
               std::mutex *compQueueMtx,
               std::condition_variable compQueueCV);*/
void initThreadPool(int numofthreads);
void exitThreadPool();
#else
#endif
