
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 512

__global__ void vecAdd(int *_a, int *_b, int *_c) {
	int tID = threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int main(void)
{

	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	int memSize = sizeof(int)*NUM_DATA; // 여기서 *는 곱하기
	printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize); 

	a = new int[NUM_DATA]; memset(a, 0, memSize); // void * memset(void * ptr, int value, size_t num); => ptr: 채우고자 하는 메모리의 시작 포인터(시작주소) value : 메모리에 채우고자 하는 값. num : 채우고자 하는 바이트의 수
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	
	// Input 생성
	for (int i = 0; i < NUM_DATA;i++) {
		a[i] = rand() % 10; 
		b[i] = rand() % 10;
	}

	// Device memory allocation
	cudaMalloc(&d_a, memSize); // "d_" 는 관습적인 prefix
	cudaMalloc(&d_b, memSize);
	cudaMalloc(&d_c, memSize);

	// Send input data from host to device
	cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

	// Kernel call
	vecAdd << <1, NUM_DATA >> > (d_a, d_b, d_c);

	// Send result from device to host
	cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

	// check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if ((a[i] + b[i]) != c[i]) {
			printf("[%d] The results is not matched! (%d, %d)\n", i, a[i] + b[i] + c[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	// Release
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	delete[] a; delete[] b; delete[] c;


	return 0;
}