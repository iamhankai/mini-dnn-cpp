#include "Util.h"

void printDeviceInfo()
{
	cudaDeviceProp devProv;
  CHECK(cudaGetDeviceProperties(&devProv, 0));
  printf("**********GPU info**********\n");
  printf("Name: %s\n", devProv.name);
  printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
  printf("Num SMs: %d\n", devProv.multiProcessorCount);
  printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
  printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
  printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
  printf("CMEM: %lu bytes\n", devProv.totalConstMem);
  printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
  printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
  printf("****************************\n");
}