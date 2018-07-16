#ifndef SGD_H
#define SGD_H

#include <Eigen/Core>
#include "utils.h"

class SGD {
private:
	float lr;
	float decay;

public:
	SGD() : lr(0.01), decay(0.0001) {}
	SGD(float lr, float decay) : lr(lr), decay(decay) {}
	
	void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
};

#endif /* SGD_H */