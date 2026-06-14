#ifndef PERCEPTRON_IMAGENES_MNIST_LOADER_H
#define PERCEPTRON_IMAGENES_MNIST_LOADER_H

#pragma once

#include <vector>
#include <string>

using namespace std;

struct MNIST_Data
{
    std::vector<float> images;
    std::vector<int> labels;

    int imageCount;
    int imageSize;
};

MNIST_Data loadMNIST( const string& imagesPath, const string& labelsPath, int limit);

#endif //PERCEPTRON_IMAGENES_MNIST_LOADER_H