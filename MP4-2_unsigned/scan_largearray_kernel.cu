#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>
#include <stdio.h>
#include <iostream>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define TILE_SIZE 1024
// You can use any other block size you wish.
#define BLOCK_SIZE 256

unsigned int** block_sum;
unsigned int** block_sum_sum;
unsigned int* sizes;
// Host Helper Functions (allocate your own data structure...)



// Device Functions

void preallocBlockSums(unsigned int num_elements)
{
    int n = num_elements;
    unsigned int i = 1;
    
    if(n%(2*BLOCK_SIZE) == 0)
	n = n/(2*BLOCK_SIZE);
    else
        n = n/(2*BLOCK_SIZE) + 1;
    while(n > 1)
    {
	i++;
	if(n%(2*BLOCK_SIZE) == 0)
	    n = n/(2*BLOCK_SIZE);
	else
	    n = n/(2*BLOCK_SIZE) + 1;	
    }
    printf("Number of stages %d\n",i);
    printf("n after calculations %d\n",n);

    block_sum = (unsigned int**)malloc(sizeof(unsigned int*)*i);
    block_sum_sum = (unsigned int**)malloc(sizeof(unsigned int*)*i);
    sizes = (unsigned int*)malloc(sizeof(unsigned int)*i);

    n = num_elements;
    i = 0;
    if(n%(2*BLOCK_SIZE) == 0)
	n = n/(2*BLOCK_SIZE);
    else
	n = n/(2*BLOCK_SIZE) + 1;
    while(n > 1)
    {

	cudaMalloc((void**)&(block_sum[i]), n * sizeof(unsigned int));
	cudaMalloc((void**)&(block_sum_sum[i]), n * sizeof(unsigned int));
	sizes[i] = n;
	i++;

	if(n%(2*BLOCK_SIZE) == 0)
	    n = n/(2*BLOCK_SIZE);
	else
	    n = n/(2*BLOCK_SIZE) + 1;
    }

    cudaMalloc((void**)&(block_sum[i]), n * sizeof(float));
    cudaMalloc((void**)&(block_sum_sum[i]), n * sizeof(float));
    sizes[i] = n;

    printf("Sizes of i %d\n",sizes[i]);
  
}

// Kernel Functions
__global__ void add_block_sum(unsigned int *output,unsigned int *input,int num_elements)
{
    if(blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 + 1 < num_elements)
    {
        output[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 + 1] += input[blockIdx.x];
	output[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2] += input[blockIdx.x];
    }
    else if (blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 < num_elements)
    {
	output[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2] += input[blockIdx.x];
    }

}
__global__ void parallel_prefix_scan(unsigned int *outArray, unsigned int *inArray,unsigned int *block_sum, int n)
{

    __shared__ unsigned int scan_array[2*BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int offset = 2*blockIdx.x*blockDim.x;

    //Loading data into shared memory
    if(offset + t < n)   
    	scan_array[t] = inArray[offset + t];
    else
	scan_array[t] = 0;
    if(offset + blockDim.x + t < n)
    	scan_array[blockDim.x + t] = inArray[offset + blockDim.x + t];
    else
	scan_array[blockDim.x + t] = 0;

    __syncthreads();

    int stride = 1;
    while(stride <= BLOCK_SIZE)
    {
        int index = (t+1)*stride*2 - 1;
        if(index < 2*BLOCK_SIZE)
            scan_array[index] += scan_array[index-stride];
        stride = stride*2;
        __syncthreads();
    }

    if (threadIdx.x==0)	 
    {	
	block_sum[blockIdx.x] = scan_array[2*blockDim.x-1];
        scan_array[2*blockDim.x-1] = 0;
    }


    stride = BLOCK_SIZE; 
    while(stride > 0) 
    {
       int index = (threadIdx.x+1)*stride*2 - 1;
       if(index < 2* BLOCK_SIZE) 
       {
          unsigned int temp = scan_array[index];
          scan_array[index] += scan_array[index-stride]; 
          scan_array[index-stride] = temp; 
       } 
       stride = stride / 2;
       __syncthreads(); 
    } 

    if(offset + t < n)
        outArray[offset + t] = scan_array[t];
    
    if(offset + blockDim.x + t < n)
	outArray[offset + blockDim.x + t] = scan_array[blockDim.x + t];

}

// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, unsigned int numElements)
{
    int i = 0;
    dim3 block, grid;
    block.x = BLOCK_SIZE;
    block.y = 1;
    block.z = 1;
    if(numElements%(2*BLOCK_SIZE) == 0)
        grid.x = numElements/(2*BLOCK_SIZE);
    else
        grid.x = numElements/(2*BLOCK_SIZE) + 1;
    grid.y = 1;
    grid.z = 1;
    std::cout<<"Number of blocks "<<grid.x<<std::endl;

    parallel_prefix_scan<<<grid,block>>>(outArray, inArray, block_sum[i], numElements);
    

    while(grid.x > 1)
    {
        if(grid.x%(2*BLOCK_SIZE) == 0)
            grid.x = grid.x/(2*BLOCK_SIZE);
        else
            grid.x = grid.x/(2*BLOCK_SIZE) + 1;

	parallel_prefix_scan<<<grid,block>>>(block_sum_sum[i],block_sum[i], block_sum[i+1], sizes[i]);
	i++;
    }
    printf("Number of runs %d \n",i);
   
    for(int j = i-1;j >=0 ; j--)
    {
	grid.x = sizes[j+1];
	add_block_sum<<<grid, block>>>(block_sum_sum[j], block_sum_sum[j+1], sizes[j]);
    }

    if(numElements%(2*BLOCK_SIZE) == 0)
        grid.x = numElements/(2*BLOCK_SIZE);
    else
        grid.x = numElements/(2*BLOCK_SIZE) + 1;  

    add_block_sum<<<grid, block>>>(outArray, block_sum_sum[0], numElements);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
