#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Core>
#include "utils.h"
#include "sgd.h"

class Layer {
public:
	virtual ~Layer() {}

	virtual void forward(const Matrix& bottom) = 0;
	virtual void backward(const Matrix& bottom, const Matrix& grad_top) = 0;
	virtual const Matrix& output() = 0;
	virtual const Matrix& back_gradient() = 0;
	virtual void update(SGD& opt) = 0;
};

#endif /* LAYER_H */
