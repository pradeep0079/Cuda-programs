
#include<stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BS 8
#define N 10

void print(int *A, int n){
    for(int i = 0; i < n; i++){
        printf("%d ", A[i]);
    }
    printf("\\n");
}

__global__ void increment(int *A, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) A[i]++;
}

int main (){
    int threadsPerBlock, blocksPerGrid, n, *A, *dA;
    n = N;
    threadsPerBlock = BS;
    blocksPerGrid = (n + BS - 1) / BS;
    A = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++){
        A[i] = i * 10; // 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
    }

    cudaMalloc((void **)&dA, n * sizeof(int));
    cudaMemcpy(dA, A, n * sizeof(int), cudaMemcpyHostToDevice);
    increment<<<blocksPerGrid, threadsPerBlock>>>(dA, n);

    cudaMemcpy(A, dA, n * sizeof(int), cudaMemcpyDeviceToHost);
    print(A, n);
    cudaFree(dA);
    free(A);
    return 0;
}





