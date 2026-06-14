#include "gpu_mlp.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <iostream>




//ini pesos-----------------------------------------------------------------------------------------------------
float randomNormal()
{
    float u1 =
        ((float)rand() + 1.0f)
        /
        ((float)RAND_MAX + 1.0f);

    float u2 =
        ((float)rand() + 1.0f)
        /
        ((float)RAND_MAX + 1.0f);

    return
        sqrtf(-2.0f * logf(u1))
        *
        cosf(2.0f * 3.1415926535f * u2);
}


















//GPU_MLP-ESTRUCTURA----------------------------------------------------------------------------
//constructur
GPUMLP::GPUMLP()
    :
    hiddenLayer1(
    INPUT_SIZE,
    HIDDEN_LAYER_1_SIZE
    ),
    hiddenLayer2(
    HIDDEN_LAYER_1_SIZE,
    HIDDEN_LAYER_2_SIZE
    ),
    outputLayer(
    HIDDEN_LAYER_2_SIZE,
    OUTPUT_SIZE
    ) {


    //INICIALIZACION DE BIASES
    for(int i = 0;i < HIDDEN_LAYER_1_SIZE;i++)
    {
        hiddenLayer1.biases[i] = 0.0f;
    }

    for(int i = 0;i < HIDDEN_LAYER_2_SIZE;i++)
    {
        hiddenLayer2.biases[i] = 0.0f;
    }

    for(int i = 0;i < OUTPUT_SIZE;i++)
    {
        outputLayer.biases[i] = 0.0f;
    }

    //INICIALIZACION DE PESOS (He)
    float stddev1 =
        sqrtf(2.0f /INPUT_SIZE);
    //HL1
    for(int neuron = 0;neuron < HIDDEN_LAYER_1_SIZE;neuron++)
    {
        for(int input = 0;
            input < INPUT_SIZE;
            input++){
            hiddenLayer1.weights[neuron * INPUT_SIZE +input]=randomNormal() *stddev1;
        }
    }

    //HL2
    float stddev2 =
        sqrtf(2.0f /HIDDEN_LAYER_1_SIZE);

    for(int neuron = 0;neuron < HIDDEN_LAYER_2_SIZE;neuron++) {
        for(int input = 0;input < HIDDEN_LAYER_1_SIZE;input++)
            {
            hiddenLayer2.weights[neuron *HIDDEN_LAYER_1_SIZE +input]=randomNormal() *stddev2;
            }
    }


    //OL
    float stddev3 =sqrtf(2.0f /HIDDEN_LAYER_2_SIZE);

    for(int neuron = 0;neuron < OUTPUT_SIZE;neuron++) {
        for(int input = 0;input < HIDDEN_LAYER_2_SIZE;input++) {
            outputLayer.weights[neuron *HIDDEN_LAYER_2_SIZE +input]=
                randomNormal() *stddev3;
            }
    }




    //RESERVA DE MEMORIAS
    hiddenLayer1.allocateGPU();
    hiddenLayer2.allocateGPU();
    outputLayer.allocateGPU();

    //BUFFERS DE FORWARD--------------------------------
    cudaMalloc(
        &d_hiddenLayer1Output,
        MAX_BATCH_SIZE *HIDDEN_LAYER_1_SIZE *sizeof(float)
    );

    cudaMalloc(
        &d_hiddenLayer2Output,
        MAX_BATCH_SIZE *HIDDEN_LAYER_2_SIZE *sizeof(float)
    );

    cudaMalloc(
        &d_output,
        MAX_BATCH_SIZE *OUTPUT_SIZE *sizeof(float)
    );


    //waaa
    //BUFFERS DE MINIBTACH---------------------
    cudaMalloc(
    &d_batchInput,
    MAX_BATCH_SIZE *INPUT_SIZE *sizeof(float)
    );

    cudaMalloc(
        &d_batchLabels,MAX_BATCH_SIZE *sizeof(int)
    );

    //BUFFERS DE BACKPROPAGATION---------------------
    cudaMalloc(
        &d_target,
MAX_BATCH_SIZE *OUTPUT_SIZE *sizeof(float)
    );

    cudaMalloc(
        &d_outputError,
        MAX_BATCH_SIZE *OUTPUT_SIZE *sizeof(float)
    );

    cudaMalloc(
        &d_hiddenLayer1Error,
        MAX_BATCH_SIZE *HIDDEN_LAYER_1_SIZE *sizeof(float)
    );

    cudaMalloc(
        &d_hiddenLayer2Error,
        MAX_BATCH_SIZE *HIDDEN_LAYER_2_SIZE *sizeof(float)
    );

}

//destructor
GPUMLP::~GPUMLP()
{
    if(d_hiddenLayer1Output)
        cudaFree(d_hiddenLayer1Output);

    if(d_hiddenLayer2Output)
        cudaFree(d_hiddenLayer2Output);

    if(d_output)
        cudaFree(d_output);
    //mini batches
    if(d_batchInput)
        cudaFree(d_batchInput);

    if(d_batchLabels)
        cudaFree(d_batchLabels);
    //
    if(d_target)
        cudaFree(d_target);

    if(d_outputError)
        cudaFree(d_outputError);

    if(d_hiddenLayer2Error)
        cudaFree(d_hiddenLayer2Error);

    if(d_hiddenLayer1Error)
        cudaFree(d_hiddenLayer1Error);
    if(d_trainImages)
        cudaFree(d_trainImages);

    if(d_trainLabels)
        cudaFree(d_trainLabels);
}










//DATASET-----------------------------------------------------------------------------

//copiar dataset a GPU
void GPUMLP::uploadDataset(
    const std::vector<float>& images,
    const std::vector<int>& labels
)
{


    cudaMalloc(
        &d_trainImages,
        images.size() * sizeof(float)
    );

    cudaMemcpy(
        d_trainImages,
        images.data(),images.size() * sizeof(float),cudaMemcpyHostToDevice
    );

    cudaMalloc(
        &d_trainLabels,
        labels.size() * sizeof(int)
    );

    cudaMemcpy(
        d_trainLabels,
        labels.data(),labels.size() * sizeof(int),cudaMemcpyHostToDevice
    );
}

//copiar pesos y bias de cpu a gpu
void GPUMLP::copyToGPU()
{
    hiddenLayer1.copyToGPU();
    hiddenLayer2.copyToGPU();
    outputLayer.copyToGPU();
}














//Forward kernels------------------------------------------------------------------------------------

//Calcula activación ReLU
__global__ void denseForwardKernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int inputSize,
    int outputSize,
    int batchSize
)
{
    int neuron =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int sample =
        blockIdx.y;

    if(neuron >= outputSize)
        return;

    if(sample >= batchSize)
        return;

    float sum =
        biases[neuron];

    const float* sampleInput =
        input + sample * inputSize;

    for(int i = 0; i < inputSize; i++)
    {
        sum +=
            weights[neuron * inputSize + i]
            *
            sampleInput[i];
    }

    output[sample * outputSize +neuron] =
        sum > 0.0f?sum:0.0f;
}

//Calcula capa sin activación
__global__ void denseOutputKernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int inputSize,
    int outputSize,
    int batchSize
)
{
    int neuron =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int sample =
        blockIdx.y;

    if(neuron >= outputSize)
        return;

    if(sample >= batchSize)
        return;

    float sum =
        biases[neuron];

    const float* sampleInput =
        input +
        sample * inputSize;

    for(int i = 0;
        i < inputSize;
        i++)
    {
        sum +=weights[neuron * inputSize + i]
            *
            sampleInput[i];
    }

    output[sample * outputSize +neuron] = sum;
}


//Softmax para salida
__global__ void softmaxKernel(
    float* output,
    int batchSize
)
{
    int sample = blockIdx.x;

    if(sample >= batchSize)
        return;

    float* row =
        output + sample * OUTPUT_SIZE ;

    float maxValue = row[0];

    for(int i = 1; i < OUTPUT_SIZE ; i++)
    {
        if(row[i] > maxValue)
            maxValue = row[i];
    }

    float sumExp = 0.0f;

    for(int i = 0; i < OUTPUT_SIZE ; i++)
    {
        row[i] = expf(row[i] - maxValue);

        sumExp += row[i];
    }

    for(int i = 0; i < OUTPUT_SIZE ; i++)
    {
        row[i] /= sumExp;
    }
}














//Backpropagation Kernels---------------------------------------------------------------------------------

//Calcular error de salida (prediction - target)
__global__ void computeOutputErrorBatchKernel(
    const float* output,
    const float* target,
    float* errorOutput,
    int batchSize
)
{
    int classIdx =
        threadIdx.x;

    int sample =
        blockIdx.x;

    if(sample >= batchSize)
        return;

    if(classIdx >= OUTPUT_SIZE )
        return;

    int idx =
        sample * OUTPUT_SIZE  +classIdx;

    errorOutput[idx] =
        output[idx]-target[idx];
}



//Propaga error Output->hidden layer 2
__global__ void computeHidden2ErrorBatchKernel(
    const float* errorOutput,
    const float* hidden2,
    const float* weights3,
    float* errorHidden2,
    int batchSize
)
{
    int neuron =
        threadIdx.x;

    int sample =
        blockIdx.x;

    if(sample >= batchSize)
        return;

    if(neuron >= HIDDEN_LAYER_2_SIZE)
        return;

    float accumulatedError =
        0.0f;

    for(int k = 0;
        k < OUTPUT_SIZE;
        k++)
    {
        accumulatedError +=
            weights3[k * HIDDEN_LAYER_2_SIZE +neuron]
            *
            errorOutput[sample * OUTPUT_SIZE +k];
    }

    float derivative =
        hidden2[sample * HIDDEN_LAYER_2_SIZE +neuron] > 0.0f?1.0f:0.0f;

    errorHidden2[sample * HIDDEN_LAYER_2_SIZE +neuron] =
        accumulatedError *derivative;
}


//Propaga error Hidden layer 2->hidden layer 1

__global__ void computeHidden1ErrorBatchKernel(
    const float* errorHidden2,
    const float* hidden1,
    const float* weights2,
    float* errorHidden1,
    int batchSize
)
{
    int neuron =
        threadIdx.x;

    int sample =
        blockIdx.x;

    if(sample >= batchSize)
        return;

    if(neuron >= HIDDEN_LAYER_1_SIZE)
        return;

    float accumulatedError =
        0.0f;

    for(int h = 0;h < HIDDEN_LAYER_2_SIZE;h++)
    {
        accumulatedError +=weights2[h * HIDDEN_LAYER_1_SIZE +neuron]
            *
        errorHidden2[sample * HIDDEN_LAYER_2_SIZE +h];
    }

    float derivative =hidden1[sample * HIDDEN_LAYER_1_SIZE +neuron] > 0.0f?1.0f:0.0f;

    errorHidden1[sample * HIDDEN_LAYER_1_SIZE +neuron]=accumulatedError*derivative;
}















//Actualizar pesos kernels--------------------------------------------------------------------------


//ACTUALIZA pesos output
__global__ void updateLayer3WeightsBatchKernel(
    float* weights,
    const float* hidden2,
    const float* errorOutput,
    float learningRate,
    int batchSize
)
{
    int inputIdx =
    blockIdx.x * blockDim.x +threadIdx.x;

    int neuron =
    blockIdx.y * blockDim.y +threadIdx.y;

    if(neuron >= OUTPUT_SIZE ||inputIdx >= HIDDEN_LAYER_2_SIZE)
        return;

    float gradient =
        0.0f;

    for(int sample = 0;sample < batchSize;sample++)
    {
        gradient +=errorOutput[sample * OUTPUT_SIZE +neuron]
            *
        hidden2[sample * HIDDEN_LAYER_2_SIZE +inputIdx];
    }

    gradient /= batchSize;

    weights[neuron * HIDDEN_LAYER_2_SIZE +inputIdx]
    -=
    learningRate*gradient;


}



//ACTUALIZA bias output
__global__ void updateLayer3BiasesKernel(
    float* biases,
    const float* errorOutput,
    float learningRate,
    int batchSize
)
{
    int neuron =
        blockIdx.x *blockDim.x +threadIdx.x;

    if(neuron >= OUTPUT_SIZE)
        return;

    float gradient = 0.0f;

    for(int sample = 0;sample < batchSize;sample++)
    {
        gradient +=errorOutput[sample * OUTPUT_SIZE +neuron
];
    }

    gradient /= batchSize;

    biases[neuron]
    -=
    learningRate *gradient;
}


//ACTUALIZA pesos hidden layer 2
__global__ void updateLayer2WeightsBatchKernel(
    float* weights,
    const float* hidden1,
    const float* errorHidden2,
    float learningRate,
    int batchSize
)
{
    int inputIdx =
    blockIdx.x * blockDim.x +threadIdx.x;

    int neuron =
    blockIdx.y * blockDim.y +threadIdx.y;

    if(neuron >= HIDDEN_LAYER_2_SIZE ||inputIdx >= HIDDEN_LAYER_1_SIZE)
        return;

    float gradient = 0.0f;

    for(int sample = 0;sample < batchSize;sample++)
    {
        gradient += errorHidden2[sample * HIDDEN_LAYER_2_SIZE +neuron]
            *
        hidden1[sample * HIDDEN_LAYER_1_SIZE +inputIdx];
    }

    gradient /= batchSize;

    weights[neuron * HIDDEN_LAYER_1_SIZE +inputIdx]
    -=
    learningRate * gradient;

}


//ACTUALIZA bias hidden layer 2
__global__ void updateLayer2BiasesKernel(
    float* biases,
    const float* errorHidden2,
    float learningRate,
    int batchSize
)
{
    int neuron =
    blockIdx.x *blockDim.x +threadIdx.x;

    if(neuron >= HIDDEN_LAYER_2_SIZE)
        return;

    float gradient = 0.0f;

    for(int sample = 0;sample < batchSize;sample++)
    {
        gradient += errorHidden2[sample * HIDDEN_LAYER_2_SIZE + neuron];
    }

    gradient /= batchSize;

    biases[neuron]
    -=
    learningRate * gradient;
}


//ACTUALIZA pess hidden layer 1
__global__ void updateLayer1WeightsBatchKernel(
    float* weights,
    const float* batchInput,
    const float* errorHidden1,
    float learningRate,
    int batchSize
)
{
    int inputIdx =
    blockIdx.x * blockDim.x + threadIdx.x;

    int neuron =
    blockIdx.y * blockDim.y + threadIdx.y;

    if(neuron >= HIDDEN_LAYER_1_SIZE ||inputIdx >= INPUT_SIZE)
        return;

    float gradient =
        0.0f;

    for(int sample = 0;sample < batchSize;sample++)
    {
        gradient += errorHidden1[sample * HIDDEN_LAYER_1_SIZE +neuron]
            *
        batchInput[sample * INPUT_SIZE +inputIdx];
    }

    gradient /= batchSize;

    weights[neuron * INPUT_SIZE + inputIdx]
    -=
    learningRate * gradient;


}



//ACTUALIZA bias hidden layer 1
__global__ void updateLayer1BiasesKernel(
    float* biases,
    const float* errorHidden1,
    float learningRate,
    int batchSize
)
{
    int neuron =
        blockIdx.x *blockDim.x + threadIdx.x;

    if(neuron >= HIDDEN_LAYER_1_SIZE)
        return;

    float gradient = 0.0f;

    for(int sample = 0;sample < batchSize;sample++)
    {
        gradient += errorHidden1[sample * HIDDEN_LAYER_1_SIZE + neuron];
    }

    gradient /= batchSize;

    biases[neuron] -=
    learningRate * gradient;
}
















//TRAINING-------------------------------------------------------------------------------------

//construye vector con labels
__global__ void buildTargetBatchKernel(
    float* target,
    const int* labels,
    int batchSize
)
{
    int classIdx =
        threadIdx.x;

    int sample =
        blockIdx.x;

    if(sample >= batchSize)
        return;

    if(classIdx >= OUTPUT_SIZE)
        return;

    target[sample * OUTPUT_SIZE +classIdx]= (classIdx ==labels[sample])?1.0f:0.0f;

}















//MLP deduccion----------------------------------------------------------------------------------------------
//
//forward propagation
void GPUMLP::forwardGPU(
    const float* batchInput,
    int batchSize
) {
    dim3 block1(HIDDEN_LAYER_1_SIZE);
    dim3 grid1(1, batchSize);

    denseForwardKernel<<<grid1,block1>>>(
        batchInput,
        hiddenLayer1.d_weights,
        hiddenLayer1.d_biases,
        d_hiddenLayer1Output,
        INPUT_SIZE,
        HIDDEN_LAYER_1_SIZE,
        batchSize
    );

    dim3 block2(HIDDEN_LAYER_2_SIZE);
    dim3 grid2(1, batchSize);

    denseForwardKernel<<<grid2,block2>>>(
     d_hiddenLayer1Output,
     hiddenLayer2.d_weights,
     hiddenLayer2.d_biases,
     d_hiddenLayer2Output,
     HIDDEN_LAYER_1_SIZE,
     HIDDEN_LAYER_2_SIZE,
     batchSize
 );

    dim3 block3(16);
    dim3 grid3(
    (OUTPUT_SIZE + block3.x - 1) /
    block3.x,
    batchSize
);

    denseOutputKernel<<<grid3,block3>>>(
    d_hiddenLayer2Output,
    outputLayer.d_weights,
    outputLayer.d_biases,
    d_output,
    HIDDEN_LAYER_2_SIZE,
    OUTPUT_SIZE,
    batchSize
);
    softmaxKernel<<<batchSize,1>>>(
    d_output,
    batchSize
);
}

//probabilidades para batch
void GPUMLP::predictGPU(
    const float* input,
    float* output,
    int batchSize
)
{



    //batch
    cudaMemcpy(
    d_batchInput,
    input,batchSize * INPUT_SIZE  * sizeof(float),cudaMemcpyHostToDevice
);

    forwardGPU(d_batchInput,batchSize);

    cudaMemcpy(
    output,d_output,
    batchSize * OUTPUT_SIZE  * sizeof(float),cudaMemcpyDeviceToHost
);


}

//predice numero de una imagen
int GPUMLP::predictClass(
    const float* input
)
{
    float output[OUTPUT_SIZE];

    predictGPU(input,output,1);

    int best = 0;

    for(int i = 1; i < OUTPUT_SIZE; i++)
    {
        if(output[i] > output[best])
        {
            best = i;
        }
    }

    return best;
}











//TRAINING------------------------------------------------------------------------------------------------------

//entrenamiento en un minibatch  forward-> backpropagation + actualizacion de pesos
void GPUMLP::trainBatchFromGPU(
    int batchStart,
    int batchSize,
    float learningRate
)
{
    //  1. Mini batch------------
    const float* batchImages =
        d_trainImages +batchStart * INPUT_SIZE;

    const int* batchLabels =
        d_trainLabels + batchStart;

    cudaMemcpy(
        d_batchLabels,
        batchLabels,batchSize *sizeof(int),cudaMemcpyDeviceToDevice
    );

    //  2. Forward propagation-----------

    forwardGPU(batchImages,batchSize);


    //  3. generacion del target y error-----------
    buildTargetBatchKernel<<<batchSize,OUTPUT_SIZE>>>(
    d_target,
    d_batchLabels,
    batchSize
);


    computeOutputErrorBatchKernel<<<batchSize,OUTPUT_SIZE>>>(
    d_output,
    d_target,
    d_outputError,
    batchSize
);


    //  4.   backpropagation----------------------
    computeHidden2ErrorBatchKernel<<<batchSize,HIDDEN_LAYER_2_SIZE>>>(
    d_outputError,
    d_hiddenLayer2Output,
    outputLayer.d_weights,
    d_hiddenLayer2Error,
    batchSize
);

    computeHidden1ErrorBatchKernel<<<batchSize,HIDDEN_LAYER_1_SIZE>>>(
        d_hiddenLayer2Error,
        d_hiddenLayer1Output,
        hiddenLayer2.d_weights,
        d_hiddenLayer1Error,
        batchSize
    );



    //Actualizacion de pesos y bias(CAPA DE SALIDA)------------------
    dim3 blockLayer3(16,16);

    dim3 gridLayer3(
    (HIDDEN_LAYER_2_SIZE + blockLayer3.x - 1) /
    blockLayer3.x,

    (OUTPUT_SIZE + blockLayer3.y - 1) /
    blockLayer3.y
);

    updateLayer3WeightsBatchKernel<<<gridLayer3,blockLayer3>>>(
        outputLayer.d_weights,
        d_hiddenLayer2Output,
        d_outputError,
        learningRate,
        batchSize
    );

    updateLayer3BiasesKernel<<<1,OUTPUT_SIZE>>>(
        outputLayer.d_biases,
        d_outputError,
        learningRate,
        batchSize
    );



    //Actualizacion de pesos y bias(SEGUNDA CAPA)------------------
    dim3 blockLayer2(16,16);

    dim3 gridLayer2(
    (HIDDEN_LAYER_1_SIZE + blockLayer2.x - 1) /
    blockLayer2.x,

    (HIDDEN_LAYER_2_SIZE + blockLayer2.y - 1) /
    blockLayer2.y
);

    updateLayer2WeightsBatchKernel<<<gridLayer2,blockLayer2>>>(
        hiddenLayer2.d_weights,
        d_hiddenLayer1Output,
        d_hiddenLayer2Error,
        learningRate,
        batchSize
    );
    updateLayer2BiasesKernel<<<1,HIDDEN_LAYER_2_SIZE>>>(
        hiddenLayer2.d_biases,
        d_hiddenLayer2Error,
        learningRate,
        batchSize
    );

    //Actualizacion de pesos y bias(PRIMERA CAPA)------------------
    dim3 block(16,16);

    dim3 grid(
    (INPUT_SIZE + block.x - 1) /
    block.x,

    (HIDDEN_LAYER_1_SIZE + block.y - 1) /
    block.y
);

    updateLayer1WeightsBatchKernel<<<grid,block>>>(
        hiddenLayer1.d_weights,
        batchImages,
        d_hiddenLayer1Error,
        learningRate,
        batchSize
    );

    updateLayer1BiasesKernel<<<1,HIDDEN_LAYER_1_SIZE>>>(
    hiddenLayer1.d_biases,
    d_hiddenLayer1Error,
    learningRate,
    batchSize
);


    cudaDeviceSynchronize();

}








