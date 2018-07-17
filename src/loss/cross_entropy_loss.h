#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "../loss.h"

class CrossEntropy: public Loss {
public:
	void evaluate(const Matrix& pred, const Matrix& target);
};

#endif /* CROSS_ENTROPY_LOSS_H */