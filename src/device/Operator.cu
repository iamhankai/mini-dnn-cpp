#include "Operator.h"

// index = c*n_row + r

// A = (n, m)   B = (m, l)
//tiled matrix multiplication
__global__ void tiled_matrixMul_kernel(float *res, float *A, float *B, int n, int m, int l) {
  __shared__ float tile1[BLOCK_WIDTH * BLOCK_HEIGHT];
  __shared__ float tile2[BLOCK_WIDTH * BLOCK_HEIGHT];
  int out_row = blockDim.y * blockIdx.y + threadIdx.y;
  int out_col = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0;
  for (int i = 0; i < (m + blockDim.x - 1) / blockDim.x; ++i) {
    int weight_idx = (i * blockDim.x + threadIdx.x) * n + out_row;
    int in_idx = out_col * m + i * blockDim.y + threadIdx.y;
    if (out_row < n && i * blockDim.x + threadIdx.x < m)
      tile1[threadIdx.y * blockDim.x + threadIdx.x] = A[weight_idx];
    else
      tile1[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    if (i * blockDim.y + threadIdx.y < m && out_col < l)
      tile2[threadIdx.y * blockDim.x + threadIdx.x] = B[in_idx];
    else
      tile2[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    //waiting until all cells in SMEM are assigned
    __syncthreads();
    for (int j = 0; j < blockDim.x; ++j) {
      sum += tile1[threadIdx.y * blockDim.x + j] * tile2[j * blockDim.x + threadIdx.x];
    }
    //waiting until all values in SMEM are processed
    __syncthreads();
  }
  if (out_row < n && out_col < l)
    res[out_col * n + out_row] = sum;
}

__global__ void matrixMul_kernel(float *res, float *A, float *B, int n, int m, int l) {
  int out_row = blockDim.y * blockIdx.y + threadIdx.y;
  int out_col = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0;
  if (out_row < n && out_col < l) {
    for (int i = 0; i < m; ++i) {
      sum += A[i * n + out_row] * B[out_col * m + i];

    }
    res[out_col * n + out_row] = sum;
  }
}

__global__ void matrixColwiseAddVec_kernel(float *des, float *vec, int n, int m) {
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_row < n && out_col < m) {
    des[out_col * n + out_row] += vec[out_row];
  }
}

__global__ void matrixRowwiseAddVec_kernel(float* des, float* vec, int n, int m) {
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_row < n && out_col < m) {
    des[out_col * n + out_row] += vec[out_col];
  }
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
  dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid_size((l + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
  tiled_matrixMul_kernel<<<grid_size, block_size>>>(d_res, d_A, d_B, n, m, l);
  // matrixMul_kernel<<<grid_size, block_size>>>(d_res, d_A, d_B, n, m, l);
  //data transfer from device back to host
  CHECK(cudaMemcpy(res, d_res, res_size, cudaMemcpyDeviceToHost));
  //free dev memory
  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_res));
}

// des = (n, m) vec = (n)
void dev_matrixColwiseAddVec(float *des, float *vec, int n, int m) {
  float* d_des = nullptr;
  float* d_vec = nullptr;
  //allocate dev memory
  CHECK(cudaMalloc(&d_des, sizeof(float) * n * m));
  CHECK(cudaMalloc(&d_vec, sizeof(float) * n));
  //data transfer from host to device
  CHECK(cudaMemcpy(d_des, des, sizeof(float) * n * m, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vec, vec, sizeof(float) * n, cudaMemcpyHostToDevice));
  //call kernel
  //default block size: 32 x 32
  dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid_size((m + block_size.x -1) / block_size.x, (n + block_size.y - 1) / block_size.y);
  matrixColwiseAddVec_kernel<<<grid_size, block_size>>>(d_des, d_vec, n , m);
  //data transfer from device back to host
  CHECK(cudaMemcpy(des, d_des, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
  //free dev memory
  CHECK(cudaFree(d_des));
  CHECK(cudaFree(d_vec));
}

// des = (n, m) vec = (m)
void dev_matrixRowwiseAddVec(float *des, float *vec, int n, int m) {
  float* d_des = nullptr;
  float* d_vec = nullptr;
  //allocate dev memory
  CHECK(cudaMalloc(&d_des, sizeof(float) * n * m));
  CHECK(cudaMalloc(&d_vec, sizeof(float) * m));
  //data transfer from host to device
  CHECK(cudaMemcpy(d_des, des, sizeof(float) * n * m, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vec, vec, sizeof(float) * m, cudaMemcpyHostToDevice));
  //call kernel
  //default block size: 32 x 32
  dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid_size((m + block_size.x -1) / block_size.x, (n + block_size.y - 1) / block_size.y);
  matrixColwiseAddVec_kernel<<<grid_size, block_size>>>(d_des, d_vec, n , m);
  //data transfer from device back to host
  CHECK(cudaMemcpy(des, d_des, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
  //free dev memory
  CHECK(cudaFree(d_des));
  CHECK(cudaFree(d_vec));
}