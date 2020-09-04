
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// kernel.cu 삭제 후 빈 cu 파일 생성, code 작성 => GPU작동 안함.
// default로 생성된 kernel.cu code를 수정, 복사, test.cu 파일 생성, 붙여넣기 => 작동함. (CUDA 10.1)
// 빨간 밑줄 해결 => 텍스트편집기->파일확장명 -> Microsoft Visual C++, cu, cuh 추가후 껏다가 재실행

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
