/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#include <iostream>
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    
    
    //Initialize matrix in shared memory
    __shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ns[TILE_WIDTH][TILE_WIDTH];
    
    //Getting thread and block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //Identify row and column of Pd
    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;
    float Pval = 0.0;


    for(int m = 0; m <= M.width/TILE_WIDTH; ++m)
    {
	if((m*TILE_WIDTH + tx) < M.width && row < P.height)
	    Ms[ty][tx] = M.elements[row*M.width + m*TILE_WIDTH + tx];
	else
	    Ms[ty][tx] = 0;
	if((m*TILE_WIDTH + ty) < N.height && col < P.width)
	    Ns[ty][tx] = N.elements[(m*TILE_WIDTH + ty)*N.width + col];
	else
	    Ns[ty][tx] = 0;

	__syncthreads();

	for(int k = 0; k<TILE_WIDTH ; ++k)
	    Pval += Ms[ty][k]*Ns[k][tx];

	__syncthreads(); 

    }
    if(row < P.height && col < P.width)
        P.elements[row*N.width + col] = Pval;
    
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
