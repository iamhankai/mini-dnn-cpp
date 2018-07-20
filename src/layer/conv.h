#ifndef CONV_H
#define CONV_H

#include "../layer.h"

class Conv: public Layer {
private:
	const int dim_in;
	int dim_out;

	int channel_in;
	int height_in;
	int width_in;
	int channel_out;
	int height_kernel;
	int width_kernel;
	int stride;
	bool padding; // have not used yet

	int height_out;
	int width_out;
	
	Matrix weight;  // weight parameter, size = channel_in*height_kernel*width_kernel*channel_out
	Vector bias;  // bias paramter, size = channel_out
	Matrix grad_weight;  // gradient w.r.t weight
	Vector grad_bias;  // gradient w.r.t bias

	std::vector<Matrix> data_cols;

	void init();

public:
	Conv(int channel_in, int height_in, int width_in, int channel_out, 
			 int height_kernel, int width_kernel, int stride = 1, bool padding = false) : 
			 dim_in(channel_in * height_in * width_in), 
			 channel_in(channel_in), height_in(height_in), width_in(width_in),
			 channel_out(channel_out), height_kernel(height_kernel), 
			 width_kernel(width_kernel), stride(stride), padding(padding)
	{
		init();
	}
	
	void forward(const Matrix& bottom);
	void backward(const Matrix& bottom, const Matrix& grad_top);
	void update(Optimizer& opt);
	void im2col(const Vector& image, Matrix& data_col);
	void col2im(const Matrix& data_col, Vector& image);
	int output_dim() { return dim_out; }
	std::vector<float> get_parameters() const;
	std::vector<float> get_derivatives() const;
	void set_parameters(const std::vector<float>& param);
};

#endif /* CONV_H */
