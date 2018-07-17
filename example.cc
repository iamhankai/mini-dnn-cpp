#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include "src/layer.h"
#include "src/loss.h"
#include "src/mnist.h"
#include "src/layer/fully_connected.h"
#include "src/layer/sigmoid.h"
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
  Layer* fc1 = new FullyConnected(dim_in, 128);
  Layer* sig1 = new Sigmoid;
  Layer* fc2 = new FullyConnected(128, 10);
  Layer* sig2 = new Sigmoid;
  // loss
  Loss* loss = new MSE;
  SGD opt(0.01, 0.0001);
  const int n_epoch = 5;
  for (int epoch = 0; epoch < n_epoch; epoch ++) {
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in, 
                                        std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1, 
                                        std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      fc1->forward(x_batch);
      sig1->forward(fc1->output());
      fc2->forward(sig1->output());
      sig2->forward(fc2->output());
      loss->evaluate(sig2->output(), target_batch);
      
      int ith_batch = start_idx / batch_size;
      if((ith_batch % 10) == 0){
        std::cout << ith_batch << "-th batch, loss: " << loss->output() << std::endl;
      }

      sig2->backward(fc2->output(), loss->back_gradient());
      fc2->backward(sig1->output(), sig2->back_gradient());
      sig1->backward(fc1->output(), fc2->back_gradient());
      fc1->backward(x_batch, sig1->back_gradient());
      fc2->update(opt);
      fc1->update(opt);
    }
    // test
    fc1->forward(dataset.test_data);
    sig1->forward(fc1->output());
    fc2->forward(sig1->output());
    sig2->forward(fc2->output());
    float acc = compute_accuracy(sig2->output(), dataset.test_labels);
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
  }
  return 0;
}

