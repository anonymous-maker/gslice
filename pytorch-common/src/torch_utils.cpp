#include "torchutils.h"
#include "common_utils.h"
#include "socket.h"

torch::Tensor convert_rawdata_to_tensor(float *rawdata, std::vector<int64_t> dims, torch::TensorOptions options) // mostly used in server code 
{
    torch::Tensor output;
    output = torch::from_blob(rawdata, torch::IntList(dims),options);
    return output;
}
torch::Tensor convert_rawdata_to_tensor(long *rawdata, std::vector<int64_t> dims, torch::TensorOptions options) // mostly used in server code 
{
    torch::Tensor output;
    output = torch::from_blob(rawdata, torch::IntList(dims),options);
    return output;
}



torch::Tensor convert_image_to_tensor(cv::Mat *image) {
    int n_channels = image->channels();
    int height = image->rows;
    int width = image->cols;

    int image_type = image->type();
   // Image Type must be one of CV_8U, CV_32F, CV_64F
    assert((image_type % 8 == 0) || ((image_type - 5) % 8 == 0) || ((image_type - 6) % 8 == 0));

    std::vector<int64_t> dims = {1, height, width, n_channels};
    std::vector<int64_t> permute_dims = {0, 3, 1, 2};
    cv::Mat image_proc = image->clone();

    torch::Tensor image_as_tensor;
    if (image_type % 8 == 0) {
      torch::TensorOptions options(torch::kUInt8);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image_proc.data, torch::IntList(dims), options);

    } else if ((image_type - 5) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat32);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image_proc.data, torch::IntList(dims), options);

    } else if ((image_type - 6) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat64);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image_proc.data, torch::IntList(dims), options);

    }
//printTimeStamp("END converting_N_clone_convert");

//printTimeStamp("START converting_N_permute_concat");
    image_as_tensor = image_as_tensor.permute(torch::IntList(permute_dims));
    image_as_tensor = image_as_tensor.toType(torch::kFloat32);
//printTimeStamp("END converting_N_permute_concat");
//  printTimeStamp("END converting_to_tensor");
//
  return image_as_tensor;
}

torch::Tensor convert_images_to_tensor(std::queue<cv::Mat> *images, int BATCH_SIZE) {
//printTimeStamp("START converting_to_tensor");

  int n_images = BATCH_SIZE;
  int n_channels = images->front().channels();
  int height = images->front().rows;
  int width = images->front().cols;

  int image_type = images->front().type();

  // Image Type must be one of CV_8U, CV_32F, CV_64F
  assert((image_type % 8 == 0) || ((image_type - 5) % 8 == 0) || ((image_type - 6) % 8 == 0));

  std::vector<int64_t> dims = {1, height, width, n_channels};
  std::vector<int64_t> permute_dims = {0, 3, 1, 2};

  std::vector<torch::Tensor> images_as_tensors;

  for (int i = 0; i < n_images; i++) {
    cv::Mat image = images->front().clone();
    images->pop();
    torch::Tensor image_as_tensor;
//printTimeStamp("START converting_N_clone_convert");

    if (image_type % 8 == 0) {
      torch::TensorOptions options(torch::kUInt8);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options);

    } else if ((image_type - 5) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat32);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options);

    } else if ((image_type - 6) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat64);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options);

    }
//printTimeStamp("END converting_N_clone_convert");

//printTimeStamp("START converting_N_permute_concat");
    image_as_tensor = image_as_tensor.permute(torch::IntList(permute_dims));
    image_as_tensor = image_as_tensor.toType(torch::kFloat32);
    images_as_tensors.push_back(image_as_tensor);
//printTimeStamp("END converting_N_permute_concat");
  }
  torch::Tensor output_tensor = torch::cat(images_as_tensors, 0);
//  printTimeStamp("END converting_to_tensor");
  return output_tensor;

}




torch::Tensor convert_images_to_tensor(std::queue<cv::Mat> images) {
//printTimeStamp("START converting_to_tensor");

  int n_images = images.size();
  int n_channels = images.front().channels();
  int height = images.front().rows;
  int width = images.front().cols;

  int image_type = images.front().type();

  // Image Type must be one of CV_8U, CV_32F, CV_64F
  assert((image_type % 8 == 0) || ((image_type - 5) % 8 == 0) || ((image_type - 6) % 8 == 0));

  std::vector<int64_t> dims = {1, height, width, n_channels};
  std::vector<int64_t> permute_dims = {0, 3, 1, 2};

  std::vector<torch::Tensor> images_as_tensors;

  for (int i = 0; i < n_images; i++) {
    cv::Mat image = images.front().clone();
    images.pop();
    torch::Tensor image_as_tensor;
//printTimeStamp("START converting_N_clone_convert");

    if (image_type % 8 == 0) {
      torch::TensorOptions options(torch::kUInt8);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options);

    } else if ((image_type - 5) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat32);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options);

    } else if ((image_type - 6) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat64);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options);

    }
//printTimeStamp("END converting_N_clone_convert");

//printTimeStamp("START converting_N_permute_concat");

    image_as_tensor = image_as_tensor.permute(torch::IntList(permute_dims));
    image_as_tensor = image_as_tensor.toType(torch::kFloat32);
    images_as_tensors.push_back(image_as_tensor);
//printTimeStamp("END converting_N_permute_concat");
  }

  torch::Tensor output_tensor = torch::cat(images_as_tensors, 0);
//printTimeStamp("END converting_to_tensor");
  return output_tensor;
}


// Convert a vector of images to torch Tensor
torch::Tensor convert_images_to_tensor(std::vector<cv::Mat> images) {
//printTimeStamp("START converting_to_tensor");

  int n_images = images.size();
  int n_channels = images[0].channels();
  int height = images[0].rows;
  int width = images[0].cols;

  int image_type = images[0].type();

  // Image Type must be one of CV_8U, CV_32F, CV_64F
  assert((image_type % 8 == 0) || ((image_type - 5) % 8 == 0) || ((image_type - 6) % 8 == 0));

  std::vector<int64_t> dims = {1, height, width, n_channels};
  std::vector<int64_t> permute_dims = {0, 3, 1, 2};

  std::vector<torch::Tensor> images_as_tensors;

  for (int i = 0; i != n_images; i++) {
    cv::Mat image = images[i].clone();
    torch::Tensor image_as_tensor;
//printTimeStamp("START converting_N_clone_convert");

    if (image_type % 8 == 0) {
      torch::TensorOptions options(torch::kUInt8);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options);

    } else if ((image_type - 5) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat32);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options);

    } else if ((image_type - 6) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat64);
//      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options);

    }
//printTimeStamp("END converting_N_clone_convert");

//printTimeStamp("START converting_N_permute_concat");

    image_as_tensor = image_as_tensor.permute(torch::IntList(permute_dims));
    image_as_tensor = image_as_tensor.toType(torch::kFloat32);
    images_as_tensors.push_back(image_as_tensor);
//printTimeStamp("END converting_N_permute_concat");
  }

  torch::Tensor output_tensor = torch::cat(images_as_tensors, 0);
//printTimeStamp("END converting_to_tensor");
  return output_tensor;
}


torch::Tensor convert_images_to_tensor( cv::Mat *images, int total_len)
{
   int n_images = total_len;
  int n_channels = images[0].channels();
  int height = images[0].rows;
  int width = images[0].cols;

  int image_type = images[0].type();

  // Image Type must be one of CV_8U, CV_32F, CV_64F
  assert((image_type % 8 == 0) || ((image_type - 5) % 8 == 0) || ((image_type - 6) % 8 == 0));

  std::vector<int64_t> dims = {1, height, width, n_channels};
  std::vector<int64_t> permute_dims = {0, 3, 1, 2};

  std::vector<torch::Tensor> images_as_tensors;
  for (int i = 0; i != n_images; i++) {
    cv::Mat image = images[i].clone();

    torch::Tensor image_as_tensor;
    if (image_type % 8 == 0) {
      torch::TensorOptions options(torch::kUInt8);
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
    } else if ((image_type - 5) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat32);
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
    } else if ((image_type - 6) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat64);
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
    }

    image_as_tensor = image_as_tensor.permute(torch::IntList(permute_dims));
    image_as_tensor = image_as_tensor.toType(torch::kFloat32);
    images_as_tensors.push_back(image_as_tensor);
  }

  torch::Tensor output_tensor = torch::cat(images_as_tensors, 0);
   
}

torch::Tensor convert_LV_vectors_to_tensor(float* input_data, int batch_size, int nz){
        std::vector<int64_t> dims = {batch_size, nz, 1, 1};
        torch::Tensor output_tensor;
        torch::TensorOptions options(torch::kFloat32);
        output_tensor = torch::from_blob(input_data, torch::IntList(dims), options).clone();
        return output_tensor;
}


torch::Tensor convert_1Dpyobject_to_tensor(PyObject *input, std::vector<int64_t> dims){
    assert(dims.size() == 1);
    torch::TensorOptions options(torch::kInt64);
    torch::Tensor output_tensor = torch::ones({dims[0]}, options);
    PyObject *seq = PyObject_GetIter(input);
    PyObject *item;
    int index =0;
    while((item = PyIter_Next(seq))){
        int64_t a = PyLong_AsLong(item); 
        output_tensor[index++] = a;
    }
    return output_tensor;
}



torch::Tensor convert_2Dpyobject_to_tensor(PyObject *input, std::vector<int64_t> dims){
    assert(dims.size() ==2);
    torch::TensorOptions options(torch::kInt64);
    torch::Tensor output_tensor = torch::ones({dims[0], dims[1]},options);
    int i =0;
    int j=0;
    PyObject *seq = PyObject_GetIter(input);
    PyObject *item;
    while((item = PyIter_Next(seq))){
          PyObject *seq2;
        PyObject *item2;
         seq2 = PyObject_GetIter(item);
        while(item2 = PyIter_Next(seq2)){
            int64_t a = PyLong_AsLong(item2); 
            output_tensor[i][j++] = a;
          }
        i++;          
        j=0;
    }
    return output_tensor;
}


// this function assumes all tensors have the same dimensionality
torch::Tensor concatToSingleTensor(std::vector<torch::Tensor> vecTensor){
        assert(vecTensor.size() != 0 );
        torch::Tensor output_tensor = torch::cat(vecTensor, 0);
        return output_tensor;
}

void sendTensorFloat(int socketfd, int reqID, torch::Tensor input){
            // 1. send dimension size
            SOCKET_txsize(socketfd, input.dim());
            // 2. send dimension data
            for (int i=0; i<input.dim(); i++) SOCKET_txsize(socketfd, input.size(i));
            // 3. send data size    
            SOCKET_txsize(socketfd, input.numel());
            // 4. send data
            SOCKET_send(socketfd, (char*)input.data<float>(),
                        input.numel() * sizeof(float), false);
            // 5. send task ID(purely used for debugging and evaluation)
            SOCKET_txsize(socketfd, reqID);
}

void sendTensorLong(int socketfd, int reqID, torch::Tensor input){
            // 1. send dimension size
            SOCKET_txsize(socketfd, input.dim());
            // 2. send dimension data
            for (int i=0; i<input.dim(); i++) SOCKET_txsize(socketfd, input.size(i));
            // 3. send data size    
            SOCKET_txsize(socketfd, input.numel());
            // 4. send data
            SOCKET_send(socketfd, (char*)input.data<long>(),
                        input.numel() * sizeof(long), false);
            // 5. send task ID(purely used for debugging and evaluation)
            SOCKET_txsize(socketfd, reqID);
}

