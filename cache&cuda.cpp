#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 32

__global__ void gaussian_elimination(float *matrix, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float local_matrix[BLOCK_SIZE][BLOCK_SIZE];

    if (row < n && col < n) {
        local_matrix[threadIdx.x][threadIdx.y] = matrix[row * n + col];
    }

    __syncthreads();

    if (row > k && col < n) {
        float factor = local_matrix[row - k - 1][k] / local_matrix[k][k];
        local_matrix[row - k - 1][col] -= factor * local_matrix[k][col];
    }

    __syncthreads();

    if (row < n && col < n) {
        matrix[row * n + col] = local_matrix[threadIdx.x][threadIdx.y];
    }
}

void generate_random_matrix(float *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (float)(rand() % 100);
        }
    }
}

void print_matrix(float *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    float *matrix = (float *)malloc(N * N * sizeof(float));
    generate_random_matrix(matrix, N);

    printf("Original matrix:\n");
    print_matrix(matrix, N);

    float *d_matrix;
    cudaMalloc((void **)&d_matrix, N * N * sizeof(float));
    cudaMemcpy(d_matrix, matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int k = 0; k < N - 1; k++) {
        gaussian_elimination<<<numBlocks, threadsPerBlock>>>(d_matrix, N, k);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(matrix, d_matrix, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Matrix after Gaussian elimination:\n");
    print_matrix(matrix, N);

    cudaFree(d_matrix);
    free(matrix);

    return 0;
}