#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "../loss.h"

class MSE: public Loss {
public:
	void evaluate(const Matrix& pred, const Matrix& target);
};

#endif /* MSE_LOSS_H */