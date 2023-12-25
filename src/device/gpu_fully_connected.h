#ifndef _GPU_FULLY_CONNECTED_H_
#define _GPU_FULLY_CONNECTED_H_

void fc_on_gpu(const float* in, const float* weight, float* out, const float* bias, int dim_in, int dim_out, int n_samples);

#endif
