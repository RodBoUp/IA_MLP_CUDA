// =========================
// mnist_loader.cpp
// =========================

#include "mnist_loader.h"

#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return
        (static_cast<int>(c1) << 24)
        +
        (static_cast<int>(c2) << 16)
        +
        (static_cast<int>(c3) << 8)
        +
        c4;
}

MNIST_Data loadMNIST(
    const string& imagesPath,
    const string& labelsPath,
    int limit
)
{
    ifstream imageFile(imagesPath,ios::binary);

    ifstream labelFile(labelsPath,ios::binary);

    if(!imageFile)
    {
        cerr
            << "No se pudo abrir images file\n";

        exit(1);
    }

    if(!labelFile)
    {
        cerr
            << "No se pudo abrir labels file\n";

        exit(1);
    }

    int magic = 0;
    int numImages = 0;
    int rows = 0;
    int cols = 0;

    imageFile.read(reinterpret_cast<char*>(&magic),4);

    magic = reverseInt(magic);

    imageFile.read(reinterpret_cast<char*>(&numImages),4);

    numImages = reverseInt(numImages);

    imageFile.read(reinterpret_cast<char*>(&rows),4);

    rows = reverseInt(rows);

    imageFile.read(reinterpret_cast<char*>(&cols),4);

    cols = reverseInt(cols);

    int magicLabels = 0;
    int numLabels = 0;

    labelFile.read(reinterpret_cast<char*>(&magicLabels),4);

    magicLabels = reverseInt(magicLabels);

    labelFile.read(reinterpret_cast<char*>(&numLabels),4);

    numLabels = reverseInt(numLabels);

    cout<< "Numero de imagenes: "<< numImages;

    cout<< "\nRows: "<< rows;

    cout<< "\nCols: "<< cols<< endl;

    limit =
    std::min(limit,numImages);

    MNIST_Data data;

    data.imageCount =
    limit;

    data.imageSize =
    rows * cols;

    data.images.resize(
    data.imageCount *data.imageSize
    );

    data.labels.resize(data.imageCount);

    for(int n = 0;n < data.imageCount;n++)
    {
        for(int i = 0;i < data.imageSize;i++)
        {
            unsigned char pixel = 0;

            imageFile.read(reinterpret_cast<char*>(&pixel),1);

            data.images[n * data.imageSize + i] =
            static_cast<float>(pixel)/255.0f;
        }

        unsigned char label = 0;

        labelFile.read(reinterpret_cast<char*>(&label),1);

        data.labels[n] =static_cast<int>(label);
    }

    cout
        << "Imagenes cargadas: "
        << data.imageCount
        << endl;

    return data;
}
