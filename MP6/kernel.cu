#include <stdio.h>

__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr, 
    unsigned int *csrColIdx, float *csrData, float *inVector, 
    float *outVector) {

    int row = blockDim.x*blockIdx.x + threadIdx.x;

    if(row < dim)
    {
	float dot = 0;
	int row_start = csrRowPtr[row];
	int row_end = csrRowPtr[row + 1];
	for(int j = row_start; j < row_end; j++)
	{
	    dot += csrData[j]*inVector[csrColIdx[j]];
	}
	
	outVector[row] = dot;

    }


    // INSERT KERNEL CODE HERE

}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm, 
    unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx, 
    unsigned int *jdsColIdx, float *jdsData, float* inVector,
    float *outVector) {

    int row = blockDim.x*blockIdx.x + threadIdx.x;

    if(row < dim)
    {
	float dot = 0;
	int rowIdx = jdsRowPerm[row];
	int row_length = jdsRowNNZ[row];
	for(int j = 0; j < row_length; j++)
	{
	    int data_index = jdsColStartIdx[j] + row;
	    int colIdx = jdsColIdx[data_index];
	    dot += jdsData[data_index]*inVector[colIdx];
	}
	outVector[rowIdx] = dot;
    }

    // INSERT KERNEL CODE HERE

}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float *csrData, float *inVector, float *outVector) {

    dim3 grid, block;
    block.x = 32;
    block.y = 1;
    block.z = 1;

    if(dim%block.x == 0)
	grid.x = dim/block.x;
    else
	grid.x = dim/block.x + 1;
    grid.y = 1;
    grid.z = 1;

    spmv_csr_kernel<<<grid, block>>>(dim, csrRowPtr, csrColIdx, csrData, inVector, outVector);
    

    // INSERT CODE HERE

}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ, 
    unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData, 
    float* inVector, float *outVector) {

    dim3 grid, block;
    block.x = 32;
    block.y = 1;
    block.z = 1;

    if(dim%block.x == 0)
	grid.x = dim/block.x;
    else
	grid.x = dim/block.x + 1;
    grid.y = 1;
    grid.z = 1;

    spmv_jds_kernel<<<grid, block>>>(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData, inVector, outVector);
    // INSERT CODE HERE

}






