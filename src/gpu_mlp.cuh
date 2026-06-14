#pragma once

#include "gpu_layer.h"
#include "../data/mnist_loader.h"


// Network architecture

static constexpr int INPUT_SIZE = 784;

static constexpr int HIDDEN_LAYER_1_SIZE = 256;

static constexpr int HIDDEN_LAYER_2_SIZE = 128;

static constexpr int OUTPUT_SIZE = 10;

// Training

static constexpr int MAX_BATCH_SIZE = 256;



class GPUMLP
{
private:



    GPULayer hiddenLayer1;
    GPULayer hiddenLayer2;
    GPULayer outputLayer;

    //batch actual

    float* d_batchInput = nullptr;
    int* d_batchLabels = nullptr;

    // Forward

    float* d_hiddenLayer1Output = nullptr;
    float* d_hiddenLayer2Output = nullptr;
    float* d_output = nullptr;

    // targets

    float* d_target = nullptr;

    //Backpropagation

    float* d_outputError = nullptr;
    float* d_hiddenLayer2Error = nullptr;
    float* d_hiddenLayer1Error = nullptr;

    //dataset

    float* d_trainImages = nullptr;
    int* d_trainLabels = nullptr;

public:

    GPUMLP();
    ~GPUMLP();

    void uploadDataset(
        const std::vector<float>& images,
        const std::vector<int>& labels
    );

    void copyToGPU();

    void forwardGPU(
        const float* batchInput,
        int batchSize
    );

    void predictGPU(
        const float* input,
        float* output,
        int batchSize
    );

    int predictClass(
        const float* input
    );

    void trainBatchFromGPU(
        int batchStart,
        int batchSize,
        float learningRate
    );
};