#include <iostream>
#include <iomanip> 
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <msclr\marshal_cppstd.h>
#include <ctime>
#include <mpi.h>
#pragma once

#using <mscorlib.dll>
#using <System.dll>
#using <System.Drawing.dll>
#using <System.Windows.Forms.dll>

using namespace std;
using namespace msclr::interop;

int* inputImage(int* w, int* h, System::String^ imagePath) {
    int* input;

    int OriginalImageWidth, OriginalImageHeight;

    System::Drawing::Bitmap BM(imagePath);

    OriginalImageWidth = BM.Width;
    OriginalImageHeight = BM.Height;
    *w = BM.Width;
    *h = BM.Height;
    input = new int[BM.Height * BM.Width];
    for (int i = 0; i < BM.Height; i++) {
        for (int j = 0; j < BM.Width; j++) {
            System::Drawing::Color c = BM.GetPixel(j, i);
            input[i * BM.Width + j] = ((c.R + c.B + c.G) / 3); // gray scale value equals the average of RGB values
        }
    }
    return input;
}

void createImage(int* image, int width, int height, int& index) {
    System::Drawing::Bitmap MyNewImage(width, height);

    for (int i = 0; i < MyNewImage.Height; i++) {
        for (int j = 0; j < MyNewImage.Width; j++) {
            if (image[i * width + j] < 0) {
                image[i * width + j] = 0;
            }
            if (image[i * width + j] > 255) {
                image[i * width + j] = 255;
            }
            System::Drawing::Color c = System::Drawing::Color::FromArgb(
                image[i * MyNewImage.Width + j],
                image[i * MyNewImage.Width + j],
                image[i * MyNewImage.Width + j]
            );
            MyNewImage.SetPixel(j, i, c);
        }
    }
    MyNewImage.Save("..//Data//Output//outputRes" + index + ".png");
    cout << "Result Image Saved " << index << endl;
}

void computeLocalHistogram(int* localImage, int localSize, int* localHistogram) {
    for (int i = 0; i < 256; i++)
        localHistogram[i] = 0;
    for (int i = 0; i < localSize; i++) {
        localHistogram[localImage[i]]++;
    }
}

// Sequential function
void sequentialHistogramEqualization(System::String^ imagePath, int& width, int& height, int& index) {
    int* imageData = inputImage(&width, &height, imagePath);
    int totalPixels = width * height;

    
    int* globalHistogram = new int[256]();
    for (int i = 0; i < totalPixels; i++) {
        globalHistogram[imageData[i]]++;
    }

    
    int* globalCDF = new int[256];
    globalCDF[0] = globalHistogram[0];
    for (int i = 1; i < 256; i++) {
        globalCDF[i] = globalCDF[i - 1] + globalHistogram[i];
    }
    for (int i = 0; i < 256; i++) {
        globalCDF[i] = (globalCDF[i] * 255) / totalPixels;
    }

    // Apply histogram equalization to the image
    for (int i = 0; i < totalPixels; i++) {
        imageData[i] = globalCDF[imageData[i]];
    }

    createImage(imageData, width, height, index);

    delete[] globalHistogram;
    delete[] globalCDF;
    delete[] imageData;
}

int main() {
    int ImageWidth = 4, ImageHeight = 4;

    int start_parallel, stop_parallel, parallelTime = 0;

    int totalPixels, index = 1;
    int* imageData = nullptr;
    int* localImage = nullptr;

    System::String^ imagePath;
    std::string img = "..//Data//Input//test.png";
    imagePath = marshal_as<System::String^>(img);

    start_parallel = clock();

    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);
        totalPixels = ImageWidth * ImageHeight;
    }

    // Broadcast total pixels
    MPI_Bcast(&totalPixels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int localSize = (totalPixels + size - 1) / size;
    localImage = new int[localSize];
    MPI_Scatter(imageData, localSize, MPI_INT, localImage, localSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local histogram
    int* localHistogram = new int[256]();
    computeLocalHistogram(localImage, localSize, localHistogram);

    // reduce local histograms to global histogram
    int* globalHistogram = new int[256]();
    MPI_Reduce(localHistogram, globalHistogram, 256, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Compute global CDF on rank 0
    int* globalCDF = new int[256];
    if (rank == 0) {
        globalCDF[0] = globalHistogram[0];
        for (int i = 1; i < 256; i++) {
            globalCDF[i] = globalCDF[i - 1] + globalHistogram[i];
        }
        for (int i = 0; i < 256; i++) {
            globalCDF[i] = (globalCDF[i] * 255) / totalPixels;
        }
    }

    // Broadcast global CDF
    MPI_Bcast(globalCDF, 256, MPI_INT, 0, MPI_COMM_WORLD);

    // Apply histogram equalization locally
    for (int i = 0; i < localSize; i++) {
        localImage[i] = globalCDF[localImage[i]];
    }

    // Gather equalized image
    MPI_Gather(localImage, localSize, MPI_INT, imageData, localSize, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    stop_parallel = clock();
    parallelTime = (stop_parallel - start_parallel) / double(CLOCKS_PER_SEC) * 1000;

    if (rank == 0) {
        createImage(imageData, ImageWidth, ImageHeight, index);

        // Sequential Execution
        int start_s = clock();
        sequentialHistogramEqualization(imagePath, ImageWidth, ImageHeight, index);
        int stop_s = clock();


        int sequentialTime = (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
        float speedUp = (double(sequentialTime) / parallelTime);


        
        cout << "\n--------------------------------------------------" << endl;
        cout << "| Metric              | Time (ms)               |" << endl;
        cout << "--------------------------------------------------" << endl;
        cout << "| Parallel Execution  | " << setw(25) << parallelTime << " |" << endl;
        cout << "| Sequential Execution| " << setw(25) << sequentialTime << " |" << endl;
        cout << "| Speedup             | " << setw(25) << speedUp << " |" << endl;
        cout << "--------------------------------------------------\n" << endl;

        delete[] imageData;
    }

    delete[] localImage;
    delete[] localHistogram;
    delete[] globalHistogram;
    delete[] globalCDF;

    return 0;
}
