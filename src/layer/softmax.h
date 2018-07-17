#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../layer.h"

class Softmax: public Layer {
public:
	void forward(const Matrix& bottom);
	void backward(const Matrix& bottom, const Matrix& grad_top);
};

#endif /* SOFTMAX_H */