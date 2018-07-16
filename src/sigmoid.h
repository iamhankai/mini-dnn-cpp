#ifndef SIGMOID_H
#define SIGMOID_H

#include "utils.h"

class Sigmoid {
private:
	Matrix top;  // layer output
	Matrix grad_bottom;  // gradient w.r.t input

public:
	void forward(const Matrix& bottom);
	void backward(const Matrix& bottom, const Matrix& grad_top);
	const Matrix& output() { return top; }
	const Matrix& back_gradient() { return grad_bottom; }
};

#endif /* SIGMOID_H */