#ifndef SRC_LAYER_RELU_H_
#define SRC_LAYER_RELU_H_

#include "../layer.h"

class ReLU : public Layer {
 public:
  void forward(const Matrix& bottom);
  void backward(const Matrix& bottom, const Matrix& grad_top);
};

#endif  // SRC_LAYER_RELU_H_
