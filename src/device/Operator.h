#ifndef _DeviceOperator_H_
#define _DeviceOperator_H_

#include <cuda_runtime.h>
#include "Util.h"

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

// index = c*n_row + r

// A = (n, m)   B = (m, l)
void dev_matrixMul(float *res, float *A, float *B, int n, int m, int l);

// des = (n, m) vec = (n)
void dev_matrixColwiseAddVec(float *des, float *vec, int n, int m);

// des = (n, m) vec = (m)
void dev_matrixRowwiseAddVec(float *des, float *vec, int n, int m);

#endif