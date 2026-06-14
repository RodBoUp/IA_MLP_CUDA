cmake_minimum_required(VERSION 4.1)
project(
        MLP_CUDA
        LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)

add_executable(MLP_CUDA src/cuda_mlp_main.cpp
        src/gpu_layer.cu
        src/gpu_layer.h
        src/gpu_mlp.cu
        src/gpu_mlp.cuh
        data/mnist_loader.cpp
        data/mnist_loader.h
        src/gpu_mlp_errors.cu)
