#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include "src/layer.h"
#include "src/loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/layer/fully_connected.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss/mse_loss.h"

int main()
{
  // data
  MNIST dataset("../data/mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  const int batch_size = 32;
  std::cout << "mnist: " << n_train << std::endl;
  std::cout << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn;
  Layer* fc1 = new FullyConnected(dim_in, 128);
  Layer* sig1 = new Sigmoid;
  Layer* fc2 = new FullyConnected(128, 10);
  Layer* softmax = new Softmax;
  dnn.add_layer(fc1);
  dnn.add_layer(sig1);
  dnn.add_layer(fc2);
  dnn.add_layer(softmax);
  // loss
  Loss* loss = new MSE;
  dnn.add_loss(loss);
  // train & test
  SGD opt(0.01, 0.0001);
  const int n_epoch = 5;
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
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
  }
  return 0;
}

