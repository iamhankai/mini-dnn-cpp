#ifndef SRC_NETWORK_H_
#define SRC_NETWORK_H_

#include <stdlib.h>
#include <vector>
#include "./layer.h"
#include "./loss.h"
#include "./optimizer.h"
#include "./utils.h"

class Network {
 private:
  std::vector<Layer*> layers;  // layer pointers
  Loss* loss;  // loss pointer

 public:
  Network() : loss(NULL) {}
  ~Network() {
    for (int i = 0; i < layers.size(); i ++) {
      delete layers[i];
    }
    if (loss) {
      delete loss;
    }
  }

  void add_layer(Layer* layer) { layers.push_back(layer); }
  void add_loss(Loss* loss_in) { loss = loss_in; }

  void forward(const Matrix& input);
  void backward(const Matrix& input, const Matrix& target);
  void update(Optimizer& opt);

  const Matrix& output() { return layers.back()->output(); }
  float get_loss() { return loss->output(); }
  /// Get the serialized layer parameters
  std::vector<std::vector<float> > get_parameters() const;
  /// Set the layer parameters
  void set_parameters(const std::vector< std::vector<float> >& param);
  /// Get the serialized derivatives of layer parameters
  std::vector<std::vector<float> > get_derivatives() const;
  /// Debugging tool to check parameter gradients
  void check_gradient(const Matrix& input, const Matrix& target, int n_points,
                      int seed = -1);
};

#endif  // SRC_NETWORK_H_
