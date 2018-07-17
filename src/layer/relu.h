#ifndef RELU_H
#define RELU_H

#include "../layer.h"

class ReLU: public Layer {
public:
	void forward(const Matrix& bottom);
	void backward(const Matrix& bottom, const Matrix& grad_top);
};

#endif /* RELU_H */