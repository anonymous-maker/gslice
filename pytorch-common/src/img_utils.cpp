#include "img_utils.h"
#include "torchutils.h"
#include "common_utils.h"
#include <string.h>

#include <mutex>
#include <condition_variable>


ThreadPool globalPool;    
// Resize an image to a given size to
cv::Mat __resize_to_a_size(cv::Mat &image, int new_height, int new_width) {

  // get original image size
  int org_image_height = image.rows;
  int org_image_width = image.cols;

  // get image area and resized image area
  float img_area = float(org_image_height * org_image_width);
  float new_area = float(new_height * new_width);

  // resize
  cv::Mat image_scaled;
  cv::Size scale(new_width, new_height);

  if (new_area >= img_area) {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_LANCZOS4);
  } else {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_AREA);
  }

  return image_scaled;
}

// Normalize an image by subtracting mean and dividing by standard deviation
cv::Mat __normalize_mean_std(cv::Mat &image, std::vector<double> mean, std::vector<double> std) {

  // clone
  cv::Mat image_normalized = image.clone();

  // convert to float
  image_normalized.convertTo(image_normalized, CV_32FC3);

  // subtract mean
  cv::subtract(image_normalized, mean, image_normalized);

  // divide by standard deviation
  std::vector<cv::Mat> img_channels(3);
  cv::split(image_normalized, img_channels);

  img_channels[0] = img_channels[0] / std[0];
  img_channels[1] = img_channels[1] / std[1];
  img_channels[2] = img_channels[2] / std[2];

  cv::merge(img_channels, image_normalized);

  return image_normalized;  
}

// 1. Preprocess
cv::Mat preprocess(cv::Mat *image, int new_height, int new_width) 
{
    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> std = {0.229, 0.224, 0.225};

  // Clone
  cv::Mat image_proc = image->clone();

  // Convert from BGR to RGB
  cv::cvtColor(image_proc, image_proc, cv::COLOR_BGR2RGB);

  // Resize image
  image_proc = __resize_to_a_size(image_proc, new_height, new_width);

  // Convert image to float
  image_proc.convertTo(image_proc, CV_32FC3);

  // 3. Normalize to [0, 1]
  image_proc = image_proc / 255.0;

  // 4. Subtract mean and divide by std
  image_proc = __normalize_mean_std(image_proc, mean, std);

  return image_proc;
}

//does not do much.. . just returns 
unsigned char* serializeMat(cv::Mat input_mat){
//    int rows = input_mat.rows;
//    int cols = input_mat.cols;
//    int channels = input_mat.channles();
    return input_mat.data;

}
cv::Mat unserializeMat(unsigned char* serial_data){

}


void deepcopyInput(img_input* dst, img_input* src){
    dst->n_height = src->n_height;
    dst->n_width = src-> n_width;
    dst->pMat = src-> pMat;
    dst->pOutput_queue = src->pOutput_queue;
    dst->poutCV = src->poutCV;
    dst-> MAX_THREAD = src -> MAX_THREAD;
    dst-> jobID = src -> jobID;
    dst->pLock = src->pLock;
    dst->skip_resize = src->skip_resize;
}


void* preprocessInput(void* vp){
    cpu_set_t cpuset;
    int corenum = (intptr_t)vp;
    bool proceed;
    //sets up which core it will run on
    pthread_t thread=pthread_self();
    CPU_SET(corenum, &cpuset);
	 if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)!=0){
            printf("Error in setting affinity for cpu thread\n");
            exit(1);
     }
 
     //for debugging purposes;
/*
     char buffer0[10];
    snprintf(buffer0, sizeof(buffer0), "%d", corenum);
    char buffer1[50] = " START resizing_thread";
    strcat(buffer1, buffer0);
    char buffer2[50] = " END resizing_thread";
    strcat(buffer2, buffer0);
    char buffer3[50] = " START waiting for output queue";
    strcat(buffer3, buffer0);
    char buffer4[50] = " END waiting for output queue";
    strcat(buffer4, buffer0);
  */
    //wait for queue  
    while(true){
        std::unique_lock<std::mutex> lk(globalPool.inputQueueMtx);
        proceed = false;
        img_input *input;
        img_input *pTemp;
        globalPool.cv.wait(lk, []{return !globalPool.taskQueue.empty();});

/*        globalPool.threadCntMtx.lock();
        pTemp = &globalPool.taskQueue.front();
        if(globalPool.perRequstThreadCnt[pTemp->jobID] >= pTemp -> MAX_THREAD){
              globalPool.threadCntMtx.unlock();
             lk.unlock();
             continue;
        }       
        globalPool.perRequstThreadCnt[pTemp->jobID]++;
        globalPool.threadCntMtx.unlock();*/
        input = (img_input*)malloc(sizeof(img_input));
        //deep copy the contents before calling 'pop'

        deepcopyInput(input, &globalPool.taskQueue.front());

        //input = &globalPool.taskQueue.front();
        globalPool.taskQueue.pop();
        lk.unlock();
        globalPool.cv.notify_one();
        assert(input != NULL);
//printTimeStamp(buffer1);
	//printf("can you see me?\n");

	torch::Tensor  tTensor;
	if(! input->skip_resize ) {
		cv::Mat tMat  = preprocess(input->pMat,input->n_height , input->n_width );
 	       tTensor = convert_image_to_tensor(&tMat);
	}
	else   {
	//printf("can you see met too?\n");

	tTensor = convert_image_to_tensor(input->pMat);
	}

//printTimeStamp(buffer2);
//printTimeStamp(buffer3);
        input->pLock->lock();
        input->pOutput_queue->push(tTensor);
        input->pLock->unlock();
        input->poutCV->notify_one();
//printTimeStamp(buffer4);

/*        globalPool.threadCntMtx.lock();
        globalPool.perRequstThreadCnt[input->jobID]--;
        globalPool.threadCntMtx.unlock();*/
        free(input);
    }
}


pthread_t initPreprocessThread(int coreid){
    pthread_attr_t attr;
    pthread_attr_init(&attr);   
    pthread_attr_setstacksize(&attr, 1024 * 1024); 
    pthread_t tid;
    if (pthread_create(&tid, &attr, preprocessInput, (void*)coreid) != 0)   
        LOG(ERROR) << "Failed to create a request handler thread.\n";   
    return tid;

}


void initThreadPool(int numofthreads){
    for(int i =0; i < numofthreads;i++){
        globalPool.tids.push_back(initPreprocessThread(i));
    }
    globalPool.nextJobID = 0;
}

void exitThreadPool(){ // returns 1 on not exiting, 0 on successful exit
        for(int i=0; i < globalPool.tids.size(); i++) pthread_join(globalPool.tids[i], NULL);
        return;
}

void waitforOutput(std::queue<cv::Mat> *output_queue, std::condition_variable *cv, size_t total_len){
    //check whether output is ready, for now lets just busy waitforOutput    

    std::mutex tempMtx;
    std::unique_lock<std::mutex> lk(tempMtx);
    while(!(output_queue-> size() == total_len) ){
        cv->wait(lk);
    }
    lk.unlock();
    /*
    while (output_vec -> size() <total_len) {
    sleep(0.01);
    }*/
    return;

}
void waitforOutput2(int *cnt, size_t total_len, int jobID){
    //check whether output is ready, for now lets just busy waitforOutput
    while (*cnt <total_len) {
    sleep(0.01);
    }
    globalPool.perRequstThreadCnt.erase(jobID);
    return;

}

void insertTaskinQueue(img_input *input){
    //get lock 
    globalPool.inputQueueMtx.lock(); 
    globalPool.taskQueue.push(*input);
//    printf("rows : %d\n", globalPool.taskQueue.front().pMat->rows);
    //release lock 
    globalPool.inputQueueMtx.unlock(); 
     //alert other threads there is an item to process
    globalPool.cv.notify_one();
}
void insertTaskinQueue(cv::Mat &input, int imagenet_row, int imagenet_col, int ncores, int jobID,
						std::queue<torch::Tensor> *computeQueue,std::mutex *compQueueMtx, 
						std::condition_variable *compQueueCV, bool skip_resize){
    //form img_input
	img_input *temp_input = (img_input*)malloc(sizeof(img_input));
    temp_input-> pMat = new cv::Mat();
	temp_input -> n_height = imagenet_row;
  	temp_input -> n_width = imagenet_col;
  	temp_input -> MAX_THREAD = ncores;
  	temp_input -> jobID = jobID;
  	*(temp_input -> pMat) = input.clone();
  	temp_input -> pOutput_queue = computeQueue;
  	temp_input -> pLock = compQueueMtx;
  	temp_input -> poutCV = compQueueCV;
	temp_input -> skip_resize = skip_resize;
  	insertTaskinQueue(temp_input);
}

/*
torch::Tensor parallelPreprocess(std::vector<cv::Mat> &input, int new_height, int new_width,int nthreads){
        // number of threads requested should not exceed the number of threads in pool
        assert(nthreads <= globalPool.tids.size());
        // control the number of threads and nubmer of inputs each thread will compute
//        printTimeStamp("START resizing_input");
        size_t total_len = input.size();
        // variables shared by all threads
        int jobID = globalPool.nextJobID++;
        globalPool.perRequstThreadCnt[jobID] = 0;
        std::mutex perReqLock;
        std::queue<cv::Mat> *output = new std::queue<cv::Mat>(); 
        std::condition_variable outputCV;
           // slice input and store them as img_input
        
        for(int i=0; i < total_len; i++){

            img_input *temp_input = (img_input*)malloc(sizeof(img_input));
            temp_input -> n_height = new_height;
            temp_input -> n_width = new_width;
            temp_input -> MAX_THREAD = nthreads;
            temp_input -> jobID = jobID;
            temp_input -> pMat = &input[i];
            temp_input -> pOutput_queue = output;
            temp_input -> pLock = &perReqLock;
            temp_input -> poutCV = &outputCV;
            insertTaskinQueue(temp_input);
        }
        waitforOutput(output, &outputCV, total_len );
       //waitforOutput2(finish_cnt,total_len, jobID);
//        printTimeStamp("END resizing_input");
//        printTimeStamp("START converting_to_tensor");
        torch::Tensor ret = convert_images_to_tensor(*output);
//        printTimeStamp("END converting_to_tensor");
        return ret;
}
*/

torch::Tensor serialPreprocess(std::vector<cv::Mat> input, int imagenet_row, int imagenet_col){
        size_t input_size = input.size();
        std::vector<cv::Mat> resized;
        for(int j =0; j< input_size; j++){
                resized.push_back(preprocess(&input[j], imagenet_row, imagenet_col));
        }
//        printTimeStamp("START converting_to_tensor");
        torch::Tensor ret = convert_images_to_tensor(resized);
//        printTimeStamp("END converting_to_tensor");
        return ret;
}

