#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_
#define block_size 16

#include <stdio.h>
#include <iostream>
// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(float *g_data, int n)
{

    __shared__ float partialsum[2*block_size];
    
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockDim.x*blockIdx.x;

   
    if(start + t < n)   
    	partialsum[t] = g_data[start + t];
    else
	partialsum[t] = 0.0f;
    if(start + blockDim.x + t < n)
    	partialsum[blockDim.x + t] = g_data[start + blockDim.x + t];
    else
	partialsum[blockDim.x + t] = 0.0f;

    for(unsigned int stride = blockDim.x; stride >= 1; stride >>= 1)
    {
	__syncthreads();
	if(t < stride)
	    partialsum[t] += partialsum[t + stride];
    }
    
    
    g_data[blockIdx.x] = partialsum[0]; //Worked
    

//***********NOTE************************************************************************
    //This method doesn't works because each thread accesses and updates g_data[0] parallel. So each thread gets the initial value of g_data and the value returned is the value returned by the last thread.
    
//if(blockIdx.x >= 0 && blockIdx.x < n/(2*block_size))
//	g_data[0] += partialsum[0];
    
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
