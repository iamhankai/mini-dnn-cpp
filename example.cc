#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include "src/layer.h"
#include "src/loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/layer/fully_connected.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/optimizer/sgd.h"

int main()
{
  // data
  MNIST dataset("../data/mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn;
  Layer* fc1 = new FullyConnected(dim_in, 128);
  Layer* fc2 = new FullyConnected(128, 32);
  Layer* fc3 = new FullyConnected(32, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* softmax = new Softmax;
  dnn.add_layer(fc1);
  dnn.add_layer(relu1);
  dnn.add_layer(fc2);
  dnn.add_layer(relu2);
  dnn.add_layer(fc3);
  dnn.add_layer(softmax);
  // loss
  Loss* loss = new CrossEntropy;
  dnn.add_loss(loss);
  // train & test
  SGD opt(0.001, 1e-4, 0.9, true);
  const int n_epoch = 5;
  const int batch_size = 128;
  for (int epoch = 0; epoch < n_epoch; epoch ++) {
    shuffle_data(dataset.train_data, dataset.train_labels);
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in, 
                                        std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1, 
                                        std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      dnn.forward(x_batch);
      dnn.backward(x_batch, target_batch);

      int ith_batch = start_idx / batch_size;
      if((ith_batch % 50) == 0){
        std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss() << std::endl;
      }

      dnn.update(opt);
    }
    // test
    dnn.forward(dataset.test_data);
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << std::endl;
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
    std::cout << std::endl;
  }
  return 0;
}

