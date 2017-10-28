#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// Matrix multiplication kernel thread specification
__global__ void ConvolutionKernel(Matrix N, Matrix P)
{

    __shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

    //Initializing the indexes
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row_o = by*TILE_SIZE + ty;
    int col_o = bx*TILE_SIZE + tx;
 
    int n = KERNEL_SIZE/2;
    int row_i = row_o - n;
    int col_i = col_o - n;

    float output = 0.0;
    
    //Loading elemnets into the shared memory
    if(row_i >= 0 && row_i < N.height && col_i >=0 && col_i < N.width )
    {
	Ns[ty][tx] = N.elements[row_i*N.width + col_i];
    }
    else
    {
	Ns[ty][tx] = 0.0;
    }
    __syncthreads();

    
    if(ty < TILE_SIZE && tx < TILE_SIZE)
    {
	for(int i = 0; i < KERNEL_SIZE; i++)
	{
	    for(int j = 0; j < KERNEL_SIZE; j++)
	    {
		output += Mc[i*KERNEL_SIZE + j]*Ns[i+ty][j+tx];
	    }
	}
    __syncthreads();
    if(row_o < P.height && col_o < P.width )
	P.elements[row_o*P.width + col_o] = output;
    }





}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
