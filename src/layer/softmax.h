#ifndef SRC_LAYER_SOFTMAX_H_
#define SRC_LAYER_SOFTMAX_H_

#include "../layer.h"

class Softmax: public Layer {
 public:
  void forward(const Matrix& bottom);
  void backward(const Matrix& bottom, const Matrix& grad_top);
};

#endif  // SRC_LAYER_SOFTMAX_H_
