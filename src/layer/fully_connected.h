#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "../layer.h"

class FullyConnected: public Layer {
private:
	const int dim_in;
	const int dim_out;
	
	Matrix weight;  // weight parameter
	Vector bias;  // bias paramter
	Matrix grad_weight;  // gradient w.r.t weight
	Vector grad_bias;  // gradient w.r.t bias

	void init();

public:
	FullyConnected(const int dim_in, const int dim_out) : 
									dim_in(dim_in), dim_out(dim_out) {
		init();
	}
	
	void forward(const Matrix& bottom);
	void backward(const Matrix& bottom, const Matrix& grad_top);
	void update(Optimizer& opt);
	int output_dim() { return dim_out; }
	std::vector<float> get_parameters() const;
	std::vector<float> get_derivatives() const;
	void set_parameters(const std::vector<float>& param);
};

#endif /* FULLY_CONNECTED_H */
