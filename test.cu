
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// kernel.cu ���� �� �� cu ���� ����, code �ۼ� => GPU�۵� ����.
// default�� ������ kernel.cu code�� ����, ����, test.cu ���� ����, �ٿ��ֱ� => �۵���. (CUDA 10.1)
// ���� ���� �ذ� => �ؽ�Ʈ������->����Ȯ��� -> Microsoft Visual C++, cu, cuh �߰��� ���ٰ� �����

__global__ void helloCUDA(void)
{
    printf("Hello CUDA\n");
}

int main(void)
{
    printf("Hello GPU from CPU!\n");

    helloCUDA<< <1, 10>>>();

    return 0;
}
