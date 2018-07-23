#include "./sgd.h"

void SGD::update(Vector::AlignedMapType& w,
                 Vector::ConstAlignedMapType& dw) {
  // refer to SGD in PyTorch:
  // https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
  // If v is zero, initialize it
  Vector& v = v_map[dw.data()];
  if (v.size() == 0) {
    v.resize(dw.size());
    v.setZero();
  }
  // update v
  v = momentum * v + (dw + decay * w);
  // update w
  if (nesterov)
    w -= lr * (momentum * v + (dw + decay * w));
  else
    w -= lr * v;
}
