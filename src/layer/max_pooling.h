#ifndef MAX_POOLING_H
#define MAX_POOLING_H

#include "../layer.h"

class MaxPooling: public Layer {
private:
	int channel_in;
	int height_in;
	int width_in;
	int dim_in;

	int height_pool;  // pooling kernel height
	int width_pool;  // pooling kernel width
	int stride;  // pooling stride

	int channel_out;
	int height_out;
	int width_out;
	int dim_out;

	std::vector<std::vector<int> > max_idxs;  // index of max values

	void init();

public:
	MaxPooling(int channel_in, int height_in, int width_in, 
						 int height_pool, int width_pool, int stride = 1) : 
						 dim_in(channel_in * height_in * width_in), 
						 channel_in(channel_in), height_in(height_in), width_in(width_in),
						 height_pool(height_pool), width_pool(width_pool), stride(stride) 
	{
		init();
	}

	void forward(const Matrix& bottom);
	void backward(const Matrix& bottom, const Matrix& grad_top);
	int output_dim() { return dim_out; }
};

#endif /* MAX_POOLING_H */