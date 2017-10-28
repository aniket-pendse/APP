/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    //Initialize matrix in shared memory
    float Ms[TILE_WIDTH][TILE_WIDTH];
    float Ns[TILE_WIDTH][TILE_WIDTH];
    
    //Getting thread and block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //Identify row and column of Pd
    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;
    float Pval = 0.0;

    for(int m = 0; m < M.width/TILE_WIDTH; ++m)
    {
	Ms[ty][tx] = M[row*M.width + m*TILE_WIDTH + tx];
	Ns[ty][tx] = N[(m*TILE_WIDTH + ty)*M.width + col];
	__syncthreads();
	//for(int k = 0; k<TILE_WIDTH ; ++k)
	    //Pval += Ms[ty][k]*Ns[k][tx];
	__syncthreads(); 

    }
    
    P[row*M.width + col] = Pval;
monkeyt
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
