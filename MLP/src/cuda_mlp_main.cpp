#include <iostream>
#include <algorithm>
#include <vector>

#include "gpu_mlp.cuh"
#include "../data/mnist_loader.h"

using namespace std;


//config
constexpr int TRAIN_SAMPLES   = 60000;
constexpr int TEST_SAMPLES    = 10000;

constexpr int BATCH_SIZE      = 64;
constexpr int TEST_BATCH_SIZE = 256;

constexpr int EPOCHS          = 100;

constexpr float LEARNING_RATE = 0.01f;


int main()
{
    GPUMLP model;

    model.copyToGPU();

    cout << "Cargando entrenamiento...\n";

    MNIST_Data train =
        loadMNIST(
            "../data/train-images.idx3-ubyte",
            "../data/train-labels.idx1-ubyte",
            TRAIN_SAMPLES
        );

    model.uploadDataset(train.images,train.labels);

    cout << "\nCargando test...\n";

    MNIST_Data test =
        loadMNIST(
"../data/t10k-images.idx3-ubyte",
            "../data/t10k-labels.idx1-ubyte",
            TEST_SAMPLES
        );

    cout << "\nIniciando entrenamiento...\n";

    for(int epoch = 0;epoch < EPOCHS;epoch++
    )
    {
        for(int batchStart = 0;batchStart < train.imageCount;batchStart += BATCH_SIZE)
        {
            int currentBatchSize =
                min(BATCH_SIZE,train.imageCount - batchStart);

            model.trainBatchFromGPU(batchStart,currentBatchSize,LEARNING_RATE);
        }

        cout << "Epoch "<< epoch + 1<< " completada\n";
    }

    cout << "\nProbando prediccion...\n";

    vector<float> outputs(TEST_BATCH_SIZE *OUTPUT_SIZE);

    int correct = 0;

    for(int start = 0;start < test.imageCount;start += TEST_BATCH_SIZE)
    {
        int currentBatch = min(TEST_BATCH_SIZE,test.imageCount - start);

        model.predictGPU(&test.images[start *test.imageSize],outputs.data(),currentBatch);

        for(int sample = 0;sample < currentBatch;sample++)
        {
            int best = 0;

            for(int c = 1;c < OUTPUT_SIZE;c++)
            {
                if(outputs[sample *OUTPUT_SIZE +c]>outputs[sample *OUTPUT_SIZE +best])
                {
                    best = c;
                }
            }

            if(best == test.labels[start + sample])
            {
                correct++;
            }
        }
    }

    float accuracy =
        100.0f *static_cast<float>(correct)/test.imageCount;

    cout<< "\nTest Accuracy: "<< accuracy<< "%"<< endl;

    return 0;
}