#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "utils.h"

class MSE {
private:
	float loss;  // value of loss
	Matrix grad_bottom;  // gradient w.r.t input

public:
	void evaluate(const Matrix& pred, const Matrix& target);
	float output() { return loss; }
	const Matrix& back_gradient() { return grad_bottom; }
};

#endif /* MSE_LOSS_H */