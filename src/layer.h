#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Core>
#include "utils.h"
#include "sgd.h"

class Layer {
protected:
	Matrix top;  // layer output
	Matrix grad_bottom;  // gradient w.r.t input
	
public:
	virtual ~Layer() {}

	virtual void forward(const Matrix& bottom) = 0;
	virtual void backward(const Matrix& bottom, const Matrix& grad_top) = 0;
	virtual const Matrix& output() { return top; }
	virtual const Matrix& back_gradient() { return grad_bottom; }
	virtual void update(SGD& opt) {}
};

#endif /* LAYER_H */
