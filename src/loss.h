#ifndef LOSS_H
#define LOSS_H

#include "utils.h"

class Loss {
protected:
	float loss;  // value of loss
	Matrix grad_bottom;  // gradient w.r.t input

public:
	virtual ~Loss() {}

	virtual void evaluate(const Matrix& pred, const Matrix& target) = 0;
	virtual float output() { return loss; }
	virtual const Matrix& back_gradient() { return grad_bottom; }
};

#endif /* LOSS_H */