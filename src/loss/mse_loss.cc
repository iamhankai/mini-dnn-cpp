#include "./mse_loss.h"

void MSE::evaluate(const Matrix& pred, const Matrix& target) {
  int n = pred.cols();
  // forward: L = sum{ (p-y).*(p-y) } / n
  Matrix diff = pred - target;
  loss = diff.cwiseProduct(diff).sum();
  loss /= n;
  // backward: d(L)/d(p) = (p-y)*2/n
  grad_bottom = diff * 2 / n;
}
