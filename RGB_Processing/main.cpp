#include <iostream>
#include <vector>
//Thread building blocks library
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range2d.h>
#include <tbb/tick_count.h>
//Free Image library
#include <FreeImagePlus.h>
#include <math.h>


using namespace std;
using namespace tbb;

int main()
{
    int nt = task_scheduler_init::default_num_threads();
    task_scheduler_init T(nt);

    //Part 1 (Greyscale Gaussian blur): -----------DO NOT REMOVE THIS COMMENT----------------------------//

    const int size = 21;
    float sumTotal = 0;
    double kernel[size][size];
    int kernelRadius = size/2;
    float sigma = 5.5;

    //setup and load input image data set
    fipImage inputBlurImage;
    inputBlurImage.load("../Images/hedgehog.png");
    inputBlurImage.convertToFloat();

    auto blurWidth = inputBlurImage.getWidth();
    auto blurHeight = inputBlurImage.getHeight();
    const float* const inputBuffer = (float*)inputBlurImage.accessPixels();

    //setup output image array
    fipImage outputBlurImage;
    outputBlurImage = fipImage(FIT_FLOAT, blurWidth, blurHeight, 32);
    float *outputBuffer = (float*)outputBlurImage.accessPixels();


    //SEQUENTIAL
    //get system time before task
    tick_count t0 = tick_count::now();

    //kernel creation
    for(int y = -kernelRadius; y <= kernelRadius; y++) {

        for(int x = -kernelRadius; x <= kernelRadius; x++) {

            kernel[y + kernelRadius][x + kernelRadius] = 1.0f / (2.0f * float(M_PI) * sigma * sigma) * exp(-((y * y + x * x) /(2.0f * sigma * sigma)));
            sumTotal += kernel[y + kernelRadius][x + kernelRadius];
        }
    }

    //adjust kernel
    for(int y = 0; y < size; y++) {

        for(int x = 0; x < size; x++) {

            kernel[y][x] = kernel[y][x] * (1.0 / sumTotal);
        }
    }

    //apply kernel
    for(int i = kernelRadius; i < blurHeight - kernelRadius; i++) {

        for(int j = kernelRadius; j < blurWidth - kernelRadius; j++) {

            for(int x = -kernelRadius; x <= kernelRadius; x++) {

                for(int y = -kernelRadius; y <= kernelRadius; y++) {

                    outputBuffer[i * blurWidth + j] += inputBuffer[(i+x) * blurWidth + (j + y)] * kernel[x + kernelRadius][y + kernelRadius];
                }
            }
        }
    }

    //get system time after task
    tick_count t1 = tick_count::now();
    cout << "Sequential time taken = " << (t1 - t0).seconds() << " seconds" << endl;

    //PARALLEL
    //get system time before task
    tick_count t2 = tick_count::now();

    //kernel creation - step size affects outcome
   parallel_for(blocked_range2d<int, int>(-kernelRadius, kernelRadius, 2, -kernelRadius, kernelRadius, 2), [&](const blocked_range2d<int, int>& range) {

        auto y1 = range.rows().begin();
        auto y2 = range.rows().end();
        auto x1 = range.cols().begin();
        auto x2 = range.cols().end();

        for(auto y = y1; y <= y2; y++) {

            for(auto x= x1; x <= x2; x++) {

                kernel[y + kernelRadius][x + kernelRadius] = 1.0f / (2.0f * float(M_PI) * sigma * sigma) * exp(-((y * y + x * x) /(2.0f * sigma * sigma)));
                sumTotal += kernel[y + kernelRadius][x + kernelRadius];
            }
        }
    });

    //adjust kernel
    parallel_for(blocked_range2d<int, int>(0, size, 0, size), [&](const blocked_range2d<int, int>& range) {

        auto y1 = range.rows().begin();
        auto y2 = range.rows().end();
        auto x1 = range.cols().begin();
        auto x2 = range.cols().end();

        for(auto y = y1; y < y2; y++) {

            for(auto x = x1; x < x2; x++) {

                kernel[y][x] = kernel[y][x] * (1.0 / sumTotal);
            }
        }
    });

    //apply kernel
    parallel_for(blocked_range2d<int, int>(kernelRadius, blurHeight - kernelRadius, kernelRadius, blurHeight - kernelRadius), [&](const blocked_range2d<int, int>& range) {

        auto j1 = range.rows().begin();
        auto j2 = range.rows().end();
        auto i1 = range.cols().begin();
        auto i2 = range.cols().end();

        for(auto i = i1; i < i2; i++) {

            for(auto j = j1; j < j2; j++) {

                for(auto x = -kernelRadius; x <= kernelRadius; x++) {

                    for(auto y = -kernelRadius; y <= kernelRadius; y++) {

                        outputBuffer[i * blurWidth + j] += inputBuffer[(i+x) * blurWidth + (j + y)] * kernel[x + kernelRadius][y + kernelRadius];
                    }
                }
            }
        }
    });

    //get system time after task
    tick_count t3 = tick_count::now();
    cout << "Parallel time taken = " << (t3 - t2).seconds() << " seconds" << endl;

    //output image
    outputBlurImage.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    outputBlurImage.convertTo24Bits();
    outputBlurImage.save("grey_blurred.png");


    //Part 2 (Colour image processing): -----------DO NOT REMOVE THIS COMMENT----------------------------//

    // Setup Input image array
    fipImage inputImage1;
    inputImage1.load("../Images/render_1.png");

    fipImage inputImage2;
    inputImage2.load("../Images/render_2.png");

    unsigned int width = inputImage1.getWidth();
    unsigned int height = inputImage1.getHeight();

    float pixels = width*height;

    // Setup Output image array
    fipImage outputImage;
    outputImage = fipImage(FIT_BITMAP, width, height, 24);

    //2D Vector to hold the RGB colour data of an image
    vector<vector<RGBQUAD>> rgbValuesOutput;
    rgbValuesOutput.resize(height, vector<RGBQUAD>(width));

    int whitePixels = parallel_reduce(

            blocked_range2d<int, int>(0, height, 0, width), // Range to process

            0, //identity value (during reduction we have: 0 + elements in array )

            //Process sub-range setup by TBB - the lambda takes a range and a value parameter which has the value of the identity parameter
            //Computes the expression over the range, giving a result for that range
            // Main lambda to evaluate over the given range
            [&](const blocked_range2d<int, int>& range, int value)->int {

                auto y1 = range.rows().begin();
                auto y2 = range.rows().end();
                auto x1 = range.cols().begin();
                auto x2 = range.cols().end();

                RGBQUAD rgb1;  //FreeImage structure to hold RGB values of a single pixel
                RGBQUAD rgb2;

                //Extract colour data from image and store it as individual RGBQUAD elements for every pixel
                for(auto y = y1; y < y2; y++){

                    for(auto x = x1; x < x2; x++) {

                        inputImage1.getPixelColor(x, y, &rgb1); //Extract pixel(x,y) colour data and place it in rgb
                        inputImage2.getPixelColor(x, y, &rgb2);

                        rgbValuesOutput[y][x].rgbRed = rgb1.rgbRed - rgb2.rgbRed;
                        rgbValuesOutput[y][x].rgbGreen = rgb1.rgbGreen - rgb2.rgbGreen;
                        rgbValuesOutput[y][x].rgbBlue = rgb1.rgbBlue - rgb2.rgbBlue;

                        //Makes any pixels that are not black into white
                        if (rgbValuesOutput[y][x].rgbRed != 0|| rgbValuesOutput[y][x].rgbGreen != 0 || rgbValuesOutput[y][x].rgbBlue != 0) {

                            rgbValuesOutput[y][x].rgbRed = 255;
                            rgbValuesOutput[y][x].rgbGreen = 255;
                            rgbValuesOutput[y][x].rgbBlue = 255;

                            value += 1;
                        }

                        //Place the pixel colour values into output image
                        outputImage.setPixelColor(x, y, &rgbValuesOutput[y][x]);
                    }
                }
                return value;
            },

            //TBB calls this to combine the results from 2 sub-ranges (x and y) which forms part of the final result
            //Combines the results from two range computations, giving a result for the combination of the two ranges.
            //Only gets called if the parallel_reduce breaks up the operation into multiple ranges
            // Sum the results obtained from the sub-range lambda above
            [&](int x, int y)->int {
                return x + y;
            }
    );


    cout << "White Pixels ->  " << whitePixels << endl;
    cout << "Percentage of total white pixels in final image -> " << whitePixels / pixels * 100 << "%" << endl;

    //Save the processed image
    outputImage.save("RGB_processed.png");

    return 0;
}