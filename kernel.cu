
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void helloCUDA(void)
{
    printf("Hello CUDA\n");
}

int main()
{
    printf("Hello GPU from CPU!\n");

    helloCUDA<< <1, 10>>>();

    return 0;
}