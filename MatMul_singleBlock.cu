/**
This is an exmple code used in the CUDA Lecture 2 (Quick Lab. 9-2) <br>
@author : Duksu Kim
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
//#include <DS_timer.h>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_CPU_THREADS	(4)

#define ROW_SIZE (32)
#define K_SIZE   (128)
#define COL_SIZE (32)

#if K_SIZE < 129
#define USE_SHARED_VER
#endif
//#define USE_SHARED_VER_C
#define USE_BASE_KERNEL

#define	WORK_LOAD (1024)
#define MAT_SIZE_A (ROW_SIZE*K_SIZE)
#define MAT_SIZE_B (K_SIZE*COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE*COL_SIZE)

// input matrix
float A[ROW_SIZE][K_SIZE];	// m * k
float B[K_SIZE][COL_SIZE];	// k * n

// timer
//DS_timer* timer;
//void setTimer(void);
//#define TIMER_HOST		0
//#define TIMER_KERNEL	1
//#define TIMER_KERNEL_SH	2
//#define TIMER_KERNEL_SH_C	3
//#define TIMER_HtoD		4
//#define TIMER_DtoH		5
//#define NUM_TIMER		(TIMER_DtoH+1)

void genInputMatrices(void);

// output matrix
float hostC[ROW_SIZE][COL_SIZE];	// host result
float deviceC[ROW_SIZE][COL_SIZE];	// device result

#define memsetZero(_P,_type,_size) memset(_P, 0, sizeof(_type)*_size);
#define dMemAlloc(_P, _type, _size) cudaMalloc(&_P, sizeof(_type)*_size);

#ifdef USE_BASE_KERNEL
__global__ void matMul_kernel(float* _A, float* _B, float* _C)
{
	int row = threadIdx.y;
	int col = threadIdx.x;
	int index = row * blockDim.x + col;

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++)
		for (int i = 0; i < WORK_LOAD; i++)
			_C[index] += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
}

__global__ void matMul_kernel_xRow(float* _A, float* _B, float* _C)
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int index = row * blockDim.y + col;

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++)
		for (int i = 0; i < WORK_LOAD; i++)
			_C[index] += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
}

#endif


void main(void)
{
	//timer = NULL;	setTimer();

	float *dA, *dB, *dC;
	dA = dB = dC = NULL;

	memsetZero(A, float, MAT_SIZE_A);	memsetZero(B, float, MAT_SIZE_B);
	memsetZero(hostC, float, MAT_SIZE_C);	memsetZero(deviceC, float, MAT_SIZE_C);

	// device memory allocaiton
	dMemAlloc(dA, float, MAT_SIZE_A);
	dMemAlloc(dB, float, MAT_SIZE_B);
	dMemAlloc(dC, float, MAT_SIZE_C);

	// generate input matrices
	genInputMatrices();

	// Host code
	//timer->onTimer(TIMER_HOST);
	//#pragma omp parallel for num_threads(NUM_CPU_THREADS)
	for (int r = 0; r < ROW_SIZE; r++)
		for (int c = 0; c < COL_SIZE; c++)
			for (int k = 0; k < K_SIZE; k++)
				for (int i = 0; i < WORK_LOAD; i++)
					hostC[r][c] += A[r][k] * B[k][c];
	//timer->offTimer(TIMER_HOST);

	// Copy input matrices : H -> D
	//timer->onTimer(TIMER_HtoD);
	cudaMemcpy(dA, A, sizeof(float)*MAT_SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float)*MAT_SIZE_B, cudaMemcpyHostToDevice);
	//timer->offTimer(TIMER_HtoD);

	dim3 blockDim(COL_SIZE, ROW_SIZE);

#ifdef USE_BASE_KERNEL
	//// Kernel call
	//timer->onTimer(TIMER_KERNEL);

	matMul_kernel << <1, blockDim >> > (dA, dB, dC);
	cudaDeviceSynchronize();

	// For matMul_kernel_xRow
	//blockDim.x = ROW_SIZE;
	//blockDim.y = COL_SIZE;
	//matMul_kernel_xRow <<<1, blockDim >> > (dA, dB, dC);
	//cudaDeviceSynchronize();

	//timer->offTimer(TIMER_KERNEL);
#endif


	// Get back result : D -> H
	//timer->onTimer(TIMER_DtoH);
	cudaMemcpy(deviceC, dC, sizeof(float)*MAT_SIZE_C, cudaMemcpyDeviceToHost);
	//timer->onTimer(TIMER_DtoH);

	// check the results
	bool isCorrect = true;

	float *pHostC = &hostC[0][0];
	float *pDeviceC = &deviceC[0][0];

	for (int i = 0; i < MAT_SIZE_C; i++) {
		if (pHostC[i] != pDeviceC[i]) {
			printf("[%d] %.2f, %.2f\n", i, pHostC[i], pDeviceC[i]);
			isCorrect = false;
			break;
		}
	}

	if (isCorrect) printf("Result is correct!\n");
	else printf("Result is not correct!!!!!!\n");

	//timer->printTimer();
	//if (timer != NULL)
	//	delete timer;
}

void genInputMatrices(void)
{
	for (int r = 0; r < ROW_SIZE; r++)
		for (int k = 0; k < K_SIZE; k++)
			A[r][k] = rand() % 100;

	for (int k = 0; k < K_SIZE; k++)
		for (int c = 0; c < COL_SIZE; c++)
			B[k][c] = rand() % 100;

	// currently disable it for making a simple example while avoiding precision problem
	// A[r][c] += (float)(rand() % 100) / 100.0;
	// B[r][c] += (float)(rand() % 100) / 100.0;
}

//void setTimer(void)
//{
//	timer = new DS_timer(NUM_TIMER);
//
//	timer->initTimers();
//	timer->setTimerName(TIMER_HOST, "CPU code");
//	timer->setTimerName(TIMER_KERNEL, "Kernel launch");
//	timer->setTimerName(TIMER_KERNEL_SH, "Kernel launch (shared ver.)");
//	timer->setTimerName(TIMER_KERNEL_SH_C, "Kernel launch (shared ver. C)");
//	timer->setTimerName(TIMER_HtoD, "[Data transter] host->device");
//	timer->setTimerName(TIMER_DtoH, "[Data transfer] device->host");
//}
