#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

__global__ void hist_kernel(uint32_t * input, size_t height, size_t width,uint32_t bins[HISTO_HEIGHT*HISTO_WIDTH]);

uint32_t *d_input;
uint32_t *device_bins;

void preallocate_memory(uint32_t *input[],size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH], uint32_t * temp_input)
{
    //uint32_t * temp_input;
    for(int i = 0; i < height; i++)
    {
	for(int j = 0; j < width; j++)
	{
	    temp_input[i*width + j] = input[i][j];
	}
    }
    

    cudaMalloc((void**)&d_input, height*width*sizeof(uint32_t));
    cudaMemcpy(d_input, temp_input, height*width*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&device_bins, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));
    cudaMemset(device_bins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));

}

void deallocate_memory(uint32_t * host_bins, uint32_t * device_bin, size_t histo_height, size_t histo_width)
{
/* 
   for(int i = 0; i < histo_height*histo_width; i++)
    {

	//printf("Kernel elements %d \n",host_bins[i]);
	//printf("Kernel elements %d \n",device_bin[i]);
	if(device_bin[i] < 255)
	    host_bins[i] = device_bin[i];
	else
	    host_bins[i] = 255;

    }
*/

    cudaMemcpy(host_bins, device_bin, histo_height*histo_width*sizeof(uint32_t), cudaMemcpyDeviceToHost);

}

void opt_2dhisto(size_t height, size_t width)
{
    /* This function should only contain grid setup 
       code and a call to the GPU histogramming kernel. 
       Any memory allocations and transfers must be done 
       outside this function */

    cudaMemset(device_bins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));
    dim3 block, grid;
    block.x = HISTO_WIDTH*HISTO_HEIGHT;
    block.y = 1;
    block.z = 1;
    
    if(width%(HISTO_WIDTH*HISTO_HEIGHT) == 0)
	grid.x = width/(HISTO_WIDTH*HISTO_HEIGHT);
    else
	grid.x = width/(HISTO_WIDTH*HISTO_HEIGHT) + 1;

    //printf("Number of blocks %d \n",grid.x);
    grid.y = 1;
    grid.z = 1;

    hist_kernel<<<grid,block>>>(d_input, height, width, device_bins);    


}

/* Include below the implementation of any other functions you need */
__global__ void hist_kernel(uint32_t * input, size_t height, size_t width,uint32_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{
   
    __shared__ uint32_t private_histo[HISTO_HEIGHT*HISTO_WIDTH];
    
    if (threadIdx.x < HISTO_WIDTH*HISTO_HEIGHT) 
	private_histo[threadIdx.x] = 0;
    
    __syncthreads();

    int i = threadIdx.x + blockDim.x*blockIdx.x;
    
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;

    while (i < height*width) {
         atomicAdd( &(private_histo[input[i]]), 1);
         i += stride;
    }
   
    __syncthreads();

    if (threadIdx.x < HISTO_WIDTH*HISTO_HEIGHT) 
        atomicAdd( &(bins[threadIdx.x]),private_histo[threadIdx.x] );

}
//Make all uint32_t

//Copy answer to CPU kernel








