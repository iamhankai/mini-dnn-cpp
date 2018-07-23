#ifndef SRC_LOSS_MSE_LOSS_H_
#define SRC_LOSS_MSE_LOSS_H_

#include "../loss.h"

class MSE: public Loss {
 public:
  void evaluate(const Matrix& pred, const Matrix& target);
};

#endif  // SRC_LOSS_MSE_LOSS_H_
