#ifndef SGD_H
#define SGD_H

#include <Eigen/Core>
#include "utils.h"

class SGD {
private:
	float lr;

public:
	SGD() : lr(0.01) {}
	SGD(float lr) : lr(lr) {}
	
	void update(Vector::AlignedMapType& x, Vector::ConstAlignedMapType& dx);
};

#endif /* SGD_H */