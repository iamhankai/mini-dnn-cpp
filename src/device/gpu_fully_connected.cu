#include "gpu_fully_connected.h"
#include "Util.h"

//input size: dim_in x n_samples
//weight size: dim_out x dim_in
//output size: dim_out x n_samples
//output = weight * input
__global__ void fc_kernel(const float* in, const float* weight, float* out, const float* bias, int dim_in, int dim_out, int n_samples) {
  int out_row = blockDim.y * blockIdx.y + threadIdx.y;
  int out_col = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0;
  if (out_row < dim_out && out_col < n_samples) {
    for (int i = 0; i < dim_in; ++i) {
      sum += weight[out_row * dim_in + i] * in[i * n_samples + out_col];

    }
    out[out_row * n_samples + out_col] = sum + bias[out_row];
  }
}

__global__ void optimized_fc_kernel(float* in, float* weight, float* out, float* bias, int dim_in, int dim_out, int n_samples) {
  extern __shared__ float tile1[];
  extern __shared__ float tile2[];
  int out_row = blockDim.y * blockIdx.y + threadIdx.y;
  int out_col = blockDim.x * blockIdx.x + threadIdx.x;
  if (out_row < dim_out && out_col < n_samples) {
    float sum = 0;
    for (int i = 0; i < n_samples / blockDim.x + 1; ++i) {
      tile1[threadIdx.y * blockDim.x + threadIdx.x] = weight[out_row * dim_in + (i * blockDim.x) + threadIdx.x];
      tile2[threadIdx.y * blockDim.x + threadIdx.x] = in[(i * blockDim.y + threadIdx.x) * n_samples + out_col];
      for (int j = 0; j < blockDim.x; ++j) {
        sum += tile1[threadIdx.y * blockDim.x + j] * tile2[j * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }
    out[out_row * n_samples + out_col] = sum + bias[out_row];
  }
}
//implementation of fully connected layer on GPU
//default block size: 32 x 32
__host__ void fc_on_gpu(float* in, float* weight, float* out, float* bias, int dim_in, int dim_out, int n_samples) {
  float* d_in = nullptr;
  float* d_weight = nullptr;
  float* d_out = nullptr;
  float* d_bias = nullptr;
  //allocate device memory
  CHECK(cudaMalloc(&d_in, sizeof(float) * dim_in * n_samples));
  CHECK(cudaMalloc(&d_weight, sizeof(float) * dim_out * dim_in));
  CHECK(cudaMalloc(&d_out, sizeof(float) * dim_out * n_samples));
  CHECK(cudaMalloc(&d_bias, sizeof(float) * dim_out));
  //transfer data from host to device
  CHECK(cudaMemcpy(d_in, in, sizeof(float) * dim_in * n_samples, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_weight, weight, sizeof(float) * dim_out * dim_in, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_bias, bias, sizeof(float) * dim_out, cudaMemcpyHostToDevice));
  //call kernel
  dim3 block_size(32, 32);
  dim3 grid_size((n_samples + block_size.x - 1) / block_size.x, (dim_out + block_size.y - 1) / block_size.y);
  fc_kernel<<<grid_size, block_size>>>(d_in, d_weight, d_out, d_bias, dim_in, dim_out, n_samples);
  //transfer data from device to host
  CHECK(cudaMemcpy(out, d_out, sizeof(float) * dim_out * n_samples, cudaMemcpyDeviceToHost));
  //free device memory
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_weight));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_bias));
}