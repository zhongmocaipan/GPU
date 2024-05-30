#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#define N 3 // 矩阵大小
#define BLOCK_SIZE 16 // 每个块的线程数
__global__ void gaussianEliminationKernel(double* matrix, int n, int currentRow) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > currentRow && row < n) {
        double factor = matrix[row * n + currentRow] / matrix[currentRow * n + currentRow];
        for (int col = currentRow; col < n + 1; ++col) {
            matrix[row * n + col] -= factor * matrix[currentRow * n + col];
        }
    }
}
void gaussianElimination(double* h_matrix, int n) {
    double* d_matrix;
    size_t size = n * (n + 1) * sizeof(double);
    cudaMalloc((void)&d_matrix, size);
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
    for (int i = 0; i < n; ++i) {
        gaussianEliminationKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_matrix, n, i);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
}
void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
std::vector<std::vector<double>> generateRandomMatrix(int size) {
    std::vector<std::vector<double>> matrix(size, std::vector<double>(size + 1));
    srand(time(0));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j <= size; ++j) {
            matrix[i][j] = static_cast<double>(rand() % 100);
        }
    }
    return matrix;
}
int main() {
    int size = N;
    auto matrix = generateRandomMatrix(size);
    std::cout << "Original Matrix:" << std::endl;
    printMatrix(matrix);
    std::vector<double> flat_matrix(size * (size + 1));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j <= size; ++j) {
            flat_matrix[i * (size + 1) + j] = matrix[i][j];
        }
    }
    gaussianElimination(flat_matrix.data(), size);
    std::vector<std::vector<double>> result_matrix(size, std::vector<double>(size + 1));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j <= size; ++j) {
            result_matrix[i][j] = flat_matrix[i * (size + 1) + j];
        }
    }
    std::cout << "Transformed Matrix:" << std::endl;
    printMatrix(result_matrix);
    return 0;
}