#!/bin/bash
LIB_TORCH_DIR=$HOME/git/pytorch/torch
#OPENCV_DIR=$HOME/opencv/OpenCV/build
BUILD_DIR=$PWD/build

mkdir -p $BUILD_DIR
cd $BUILD_DIR
# export required environmental variables
#cmake -DCMAKE_PREFIX_PATH=$LIB_TORCH_DIR;${OPENCV_DIR} ..

#cmake -DBUILD_TIFF=ON

#cmake -DCMAKE_PREFIX_PATH=$LIB_TORCH_DIR ..
 cmake -DCMAKE_PREFIX_PATH=$LIB_TORCH_DIR -DCMAKE_BUILD_TYPE=Debug ..
#cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.2 ..

make -j 32
