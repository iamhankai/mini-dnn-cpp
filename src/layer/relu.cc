#include "./relu.h"

void ReLU::forward(const Matrix& bottom) {
  // a = z*(z>0)
  top = bottom.cwiseMax(0.0);
}

void ReLU::backward(const Matrix& bottom, const Matrix& grad_top) {
  // d(L)/d(z_i) = d(L)/d(a_i) * d(a_i)/d(z_i)
  //             = d(L)/d(a_i) * 1*(z_i>0)
  Matrix positive = (bottom.array() > 0.0).cast<float>();
  grad_bottom = grad_top.cwiseProduct(positive);
}
