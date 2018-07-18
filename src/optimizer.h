#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "utils.h"

class Optimizer {
protected:
	float lr;  // learning rate
	float decay;  // weight decay factor (default: 0)

public:
	Optimizer(float lr = 0.01, float decay = 0.0) : lr(lr), decay(decay) {}
	virtual ~Optimizer() {}
	
	virtual void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw) = 0;
};

#endif /* OPTIMIZER_H */