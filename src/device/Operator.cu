#include "Operator.h"

// index = c*n_row + r

// A = (n, m)   B = (m, l)
//tiled matrix multiplication
__global__ void matrixMul_kernel(float *res, float *A, float *B, int n, int m, int l) {
  extern __shared__ float tile1[];
  extern __shared__ float tile2[];
  int out_row = blockDim.y * blockIdx.y + threadIdx.y;
  int out_col = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0;
  for (int i = 0; i < l / blockDim.x; ++i) {
    int weight_idx = (i * blockDim.x + threadIdx.x) * n + out_row;
    int in_idx = out_col * m + i * blockDim.y + threadIdx.y;
    if (weight_idx < m * n)
      tile1[threadIdx.y * blockDim.x + threadIdx.x] = A[weight_idx];
    else
      tile1[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    if (in_idx < m * l)
      tile2[threadIdx.y * blockDim.x + threadIdx.x] = B[in_idx];
    else
      tile2[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    for (int j = 0; j < blockDim.x; ++j) {
      sum += tile1[threadIdx.y * blockDim.x + j] * tile2[j * blockDim.x + threadIdx.x];
    }
    __syncthreads();
  }
  if (out_row < n && out_col < l)
    res[out_col * n + out_row] = sum;
}

void dev_matrixMul(float *res, float *A, float *B, int n, int m, int l) {
  size_t A_size = sizeof(float) * n * m;
  size_t B_size = sizeof(float) * m * l;
  size_t res_size = sizeof(float) * n * l;
  //allocate dev memory
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_res = nullptr;
  CHECK(cudaMalloc(&d_A, A_size));
  CHECK(cudaMalloc(&d_B, B_size));
  CHECK(cudaMalloc(&d_res, res_size));
  //data transfer from host to device
  CHECK(cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, B, B_size, cudaMemcpyHostToDevice));
  //call kernel
  //default block size: 32 x 32
  dim3 block_size(32, 32);
  dim3 grid_size((l + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
  matrixMul_kernel<<<grid_size, block_size, sizeof(float) * block_size.x * block_size.y>>>(d_res, d_A, d_B, n, m, l);
  //data transfer from device back to host
  CHECK(cudaMemcpy(res, d_res, res_size, cudaMemcpyDeviceToHost));
  //free dev memory
  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_res));
}

// des = (n, m) vec = (n)
void dev_matrixColwiseAddVec(float *des, float *vec, int n, int m) {

}

// des = (n, m) vec = (m)
void dev_matrixRowwiseAddVec(float *des, float *vec, int n, int m) {

}