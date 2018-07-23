#include "./softmax.h"

void Softmax::forward(const Matrix& bottom) {
  // a = exp(z) / \sum{ exp(z) }
  top.array() = (bottom.rowwise() - bottom.colwise().maxCoeff()).array().exp();
  RowVector z_exp_sum = top.colwise().sum();  // \sum{ exp(z) }
  top.array().rowwise() /= z_exp_sum;
}

void Softmax::backward(const Matrix& bottom, const Matrix& grad_top) {
  // d(L)/d(z_i) = \sum{ d(L)/d(a_j) * d(a_j)/d(z_i) }
  // = \sum_(i!=j){ d(L)/d(a_j) * d(a_j)/d(z_i) } + d(L)/d(a_i) * d(a_i)/d(z_i)
  // = a_i * ( d(L)/d(a_i) - \sum{a_j * d(L)/d(a_j)} )
  RowVector temp_sum = top.cwiseProduct(grad_top).colwise().sum();
  grad_bottom.array() = top.array().cwiseProduct(grad_top.array().rowwise()
                                                 - temp_sum);
}
