#include "input.h"
#include <opencv2/opencv.hpp>
#include <vector> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "img_utils.h"
#include "torchutils.h"
#define SKIP_RESIZE 1
#define NZ 100 
#define IMAGENET_ROW 224
#define IMAGENET_COL 224
using namespace cv;
vector<Mat> glbMem;
vector<torch::Tensor> glbImgTensors;

torch::Tensor convertToTensor(float *input_data, int batch_size, int nz){
            return convert_LV_vectors_to_tensor(input_data, batch_size,nz);
}

float* generateLV(int batch_size, int nz){
    float* rand_data = (float *)  malloc(batch_size * nz * sizeof(float));
    assert(rand_data != NULL);
    srand (static_cast <unsigned> (time(0)));
        
    for(int i=0; i < batch_size; i++){
        for(int j = 0 ; j<nz ;j++ ){
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
               rand_data[i*nz + j] = r;
        }   
    }   
    return rand_data;
}


torch::Tensor getRandLatVec(int batch_size){ // used for dcgan
	float *rand_data;
 	rand_data=generateLV(batch_size, NZ);                                                                                                    
    torch::Tensor input = convertToTensor(rand_data, batch_size, NZ);
    return input;
}


/*torch::Tensor getRandImgTensor(){// used for img apps
	std::vector<Mat> inputImages;
	int rand_val=std::rand()%(IMG_POOL);
    inputImages.push_back(glbMem[rand_val]);                                                                                              
    torch::Tensor input; 
    if(!SKIP_RESIZE)
        input = serialPreprocess(inputImages, IMAGENET_ROW, IMAGENET_COL);
    else
        input = convert_images_to_tensor(inputImages);
	return input;
}*/
torch::Tensor getRandImgTensor(){// used for img apps
	int rand_val=std::rand()%(IMG_POOL);
    return glbImgTensors[rand_val];                                                                                              
}



int readImgData(const char *path_to_txt, int num_of_img)
{
        std::ifstream file(path_to_txt);
        std::string img_file; 
        std::vector<Mat> imgs;
		int cnt=0;
        for (int i =0; i<num_of_img;i++)
        {   
            imgs.clear();
            if ( !getline(file, img_file)) break;
//            LOG(ERROR) << "Reading " << img_file;
            Mat img;
            img= imread(img_file,  IMREAD_COLOR);
           if (img.empty())
                LOG(ERROR) << "Failed to read  " << img_file<< "\n";
#ifdef DEBUG
 /*               LOG(ERROR) << "dimensions of "<<img_file<<"\n";
                LOG(ERROR) << "size: "<<img.size()<<"\n"
                <<"row: "<<img.rows<<"\n"
                <<"column: "<<img.cols<<"\n"
                <<"channel: "<<img.channels()<<"\n";*/
#endif 

            imgs.push_back(img);
            glbImgTensors.push_back(convert_images_to_tensor(imgs));
            //glbMem.push_back(img);
		cnt++;
        }   
        if (cnt < 1) {LOG(FATAL) << "No images read!"; return 1;}
#ifdef DEBUG
    LOG(ERROR) << "read " << cnt << "images \n";
#endif
    return 0;
}

