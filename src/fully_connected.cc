#include "fully_connected.h"
#include <iostream>

void FullyConnected::init() {
	weight.resize(dim_in, dim_out);
	bias.resize(dim_out);
	grad_weight.resize(dim_in, dim_out);
	grad_bias.resize(dim_out);
	set_normal_random(weight.data(), weight.size(), 0, 0.01);
	set_normal_random(bias.data(), bias.size(), 0, 0.01);
	//std::cout << weight << std::endl;
	//std::cout << bias << std::endl;
}

void FullyConnected::forward(const Matrix& bottom) {
	// z = w' * x + b
	const int n_sample = bottom.cols();
	top.resize(dim_out, n_sample);
	top = weight.transpose() * bottom;
	top.colwise() += bias;
}

void FullyConnected::backward(const Matrix& bottom, const Matrix& grad_top) {
	const int n_sample = bottom.cols();
	// d(L)/d(w') = d(L)/d(z) * x'
	// d(L)/d(b) = \sum{ d(L)/d(z_i) }
	grad_weight = bottom * grad_top.transpose();
	grad_bias = grad_top.rowwise().sum();
	// d(L)/d(x) = w * d(L)/d(z)
	grad_bottom.resize(dim_in, n_sample);
	grad_bottom = weight * grad_top;
}

const Matrix& FullyConnected::output() {
	return top;
}

const Matrix& FullyConnected::back_gradient() {
	return grad_bottom;
}

void FullyConnected::update (SGD& opt) {
	Vector::AlignedMapType weight_vec(weight.data(), weight.size());
	Vector::AlignedMapType bias_vec(bias.data(), bias.size());
	Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
	Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

	opt.update(weight_vec, grad_weight_vec);
	opt.update(bias_vec, grad_bias_vec);
}