#ifndef __INPUT_
#define __INPUT_
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

using namespace std;
#define IMG_POOL 592 //the maximum number of image you can read

torch::Tensor getRandLatVec(int batch_size);
torch::Tensor getRandImgTensor();
int readImgData(const char *path_to_txt, int num_of_img);
#else
#endif 
