#include "gpu_layer.h"

#include <cuda_runtime.h>

GPULayer::GPULayer()
{
    inputSize = 0;
    outputSize = 0;

    weights = nullptr;
    biases = nullptr;

    d_weights = nullptr;
    d_biases = nullptr;
}

GPULayer::GPULayer(
    int inputs,
    int outputs
)
{
    inputSize = inputs;
    outputSize = outputs;

    weights = new float[inputSize * outputSize];

    biases = new float[outputSize];

    d_weights = nullptr;
    d_biases = nullptr;
}


void GPULayer::allocateGPU()
{
    cudaMalloc(
    &d_weights,inputSize *outputSize *sizeof(float)
    );

    cudaMalloc(
    &d_biases,outputSize *sizeof(float)
    );
}


void GPULayer::copyToGPU()
{
    cudaMemcpy(
    d_weights,
    weights,inputSize *outputSize *sizeof(float),cudaMemcpyHostToDevice
    );

    cudaMemcpy(
    d_biases,
    biases,outputSize *sizeof(float),cudaMemcpyHostToDevice
    );
}

void GPULayer::copyFromGPU()
{
    cudaMemcpy(
    weights
    ,d_weights,inputSize *outputSize *sizeof(float),cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
    biases,
    d_biases,outputSize *sizeof(float),cudaMemcpyDeviceToHost
    );
}


void GPULayer::freeGPU()
{
    if(d_weights)
    {
        cudaFree(d_weights);
        d_weights = nullptr;
    }

    if(d_biases)
    {
        cudaFree(d_biases);
        d_biases = nullptr;
    }
}


GPULayer::~GPULayer()
{
    delete[] weights;
    delete[] biases;

    freeGPU();
}