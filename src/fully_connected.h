#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <Eigen/Core>
#include "utils.h"
#include "sgd.h"

class FullyConnected {
private:
	const int dim_in;
	const int dim_out;

	Matrix weight;  // weight parameter
	Vector bias;  // bias paramter
	Matrix grad_weight;  // gradient w.r.t weight
	Vector grad_bias;  // gradient w.r.t bias
	Matrix top;  // layer output
	Matrix grad_bottom;  // gradient w.r.t input

public:
	FullyConnected(int dim_in, int dim_out) : dim_in(dim_in), dim_out(dim_out) {}
	void init();
	void forward(const Matrix& bottom);
	void backward(const Matrix& bottom, const Matrix& grad_top);
	const Matrix& output();
	const Matrix& back_gradient();
	void update(SGD& opt);
};

#endif /* FULLY_CONNECTED_H */
