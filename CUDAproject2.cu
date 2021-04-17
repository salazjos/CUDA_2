  /*
  Joseph Salazar
  salazjos@oregonstate.edu
  CUDAproject2.cu, Autocorrelation using CUDA
*/

// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector> 

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

//define std
using std::ofstream;
using std::ifstream;
using std::endl;
using std::string;
using std::cout;

#ifndef BLOCKSIZE
#define BLOCKSIZE		32		// number of threads per block
#endif

/*
#ifndef SIZE
#define SIZE			1*1024*1024	// array size
#endif
*/

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif

__global__ void AutoCorrelation(float *, float *, int);

int main(int argc, char* argv[ ]){

    int dev = findCudaDevice(argc, (const char **)argv);
    
    //variables
    float *h_Array = NULL;
    float *h_Sums  = NULL;
    std::vector<float> vector_data;
    ifstream fin;
    
    fin.open("signal.txt", std::ifstream::in);
    if(!fin){
        printf("cannot open file 'signal.txt'\n");
        exit(1);
    }

    while(!fin.eof()){
      float data;
      fin >> data;
      vector_data.push_back(data);
    }
    fin.close();

    int Size =(int)vector_data.at(0);
    h_Array = new float[2*Size];
    h_Sums  = new float[1*Size];
    for(int i =0 ; i < Size; i++){
      h_Array[i] = vector_data.at(i);
    }
    /*cout <<"Size is: "<<Size<<endl; 
    for(int i = 0; i < Size; i++)
        cout << h_Array[i] <<endl; 
    */

    // allocate device memory
    float *d_Array;
    float *d_Sums;

    dim3 dimsArray(2*Size, 1, 1);
    dim3 dimsSums(Size, 1, 1);

    cudaError_t status;
    status = cudaMalloc(reinterpret_cast<void **>(&d_Array), 2*Size*sizeof(float));
    checkCudaErrors(status);
    status = cudaMalloc(reinterpret_cast<void **>(&d_Sums), 1*Size*sizeof(float));
    checkCudaErrors(status);

    // copy host memory to the device
    status = cudaMemcpy(d_Array, h_Array, 2*Size*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors(status);
    status = cudaMemcpy(d_Sums, h_Sums, 1*Size*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors(status);

    // setup the execution parameters:
    dim3 threads(BLOCKSIZE, 1, 1);
    dim3 grid( Size / threads.x, 1, 1);

    // Create and start timer
    cudaDeviceSynchronize();

    // allocate CUDA events that we'll use for timing:
    cudaEvent_t start, stop;
    status = cudaEventCreate(&start);
    checkCudaErrors(status);
    status = cudaEventCreate(&stop);
    checkCudaErrors(status);

    // record the start event:
    status = cudaEventRecord(start, NULL);
    checkCudaErrors(status);

    // execute the kernel
    AutoCorrelation<<<grid,threads>>>(d_Array, d_Sums,Size);

    // record the stop event:
    status = cudaEventRecord(stop, NULL);
    checkCudaErrors(status);

    //wait for the stop event
    status = cudaEventSynchronize(stop);
    checkCudaErrors(status);

    //calculate elapsed time
    float msecTotal = 0.0f;
    status = cudaEventElapsedTime(&msecTotal, start, stop);
    checkCudaErrors(status);

    //compute peformance and print
    double secondsTotal = 0.001 * (double)msecTotal;
    double multsPerSecond =  (float)Size*Size / secondsTotal;
    double megaMultsPerSecond = multsPerSecond / 1000000.;
    printf("MegaMult/Second = %10.8lf\n", megaMultsPerSecond);

    // copy result from the device to the host:
    status = cudaMemcpy( h_Sums, d_Sums,1*Size*sizeof(float),cudaMemcpyDeviceToHost);
    checkCudaErrors(status);

    // clean up host memory:
    delete [] h_Array;
    delete [] h_Sums;

    //clean up device memory
    status = cudaFree(d_Array);
    checkCudaErrors(status);
    status = cudaFree(d_Sums);
    checkCudaErrors(status);

    return 0;

}

//Instructor provided
__global__ void AutoCorrelation(float *array, float *sums, int size){

    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int shift = gid;
    float sum = 0.0f;
	  for( int i = 0; i < size; i++ )
	  {
		  sum += array[i] * array[i + shift];
	  }
	
  sums[shift] = sum;
}
