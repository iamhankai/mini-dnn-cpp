#include "./cross_entropy_loss.h"

void CrossEntropy::evaluate(const Matrix& pred, const Matrix& target) {
  int n = pred.cols();
  const float eps = 1e-8;
  // forward: L = \sum{ -y_i*log(p_i) } / n
  loss = - (target.array().cwiseProduct((pred.array() + eps).log())).sum();
  loss /= n;
  // backward: d(L)/d(p_i) = -y_i/p_i/n
  grad_bottom = - target.array().cwiseQuotient(pred.array() + eps) / n;
}
