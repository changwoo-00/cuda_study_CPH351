
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


__global__ void checkIndex(void) {
	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n"
		, threadIdx.x, threadIdx.y, threadIdx.z
		, blockIdx.x, blockIdx.y, blockIdx.z
		, blockDim.x, blockDim.y, blockDim.z
		, gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv) {
	int nElem = 6; 

	dim3 block(3); // block(3) == block(3,1,1)
	dim3 grid((nElem + block.x - 1) / block.x); // (6+3-1)/3 = 2.xx -> 2 (usigned int) // 6개의 원소를 처리하는데 block이 몇개 필요한지 계산

	printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
	printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

	checkIndex << <grid, block >> > ();
	cudaDeviceReset();
	return(0);
}