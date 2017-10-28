/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  //Multiply the two matrices
  int row = threadIdx.y + blockIdx.y*blockDim.y;
  int col = threadIdx.x + blockIdx.x*blockDim.x;  

    if(row < M.width && col < M.height){
    float P_element = 0.0;
    //Computing matrix multiplication
    for(int k=0;k < M.width;++k)
	P_element += M.elements[row*M.width + k]*N.elements[k*M.width + col];

    P.elements[row*M.width + col] = P_element;
    }
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
