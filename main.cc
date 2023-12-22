/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
//
#include "src/parameter.h"

#include "src/device/Util.h"

int main() {
  printDeviceInfo();

  // Load Fashion MNIST dataset
  MNIST dataset("../data/mnist/");
  dataset.read();

  // Build Lenet5 model
  Network dnn;
  Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5, 1, 0, 0);
  Layer* pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
  Layer* conv2 = new Conv(6, 12, 12, 16, 5, 5, 1, 0, 0);
  Layer* pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
  Layer* fc3 = new FullyConnected(pool2->output_dim(), 120);
  Layer* fc4 = new FullyConnected(120, 84);
  Layer* fc5 = new FullyConnected(84, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* relu3 = new ReLU;
  Layer* relu4 = new ReLU;
  Layer* softmax = new Softmax;
  dnn.add_layer(conv1);
  dnn.add_layer(relu1);
  dnn.add_layer(pool1);
  dnn.add_layer(conv2);
  dnn.add_layer(relu2);
  dnn.add_layer(pool2);
  dnn.add_layer(fc3);
  dnn.add_layer(relu3);
  dnn.add_layer(fc4);
  dnn.add_layer(relu4);
  dnn.add_layer(fc5);
  dnn.add_layer(softmax);

  // load parameters
  std::vector<float> conv1Parameters = loadParametersFromFile("../parameters/conv1.txt");
  std::vector<float> conv2Parameters = loadParametersFromFile("../parameters/conv2.txt");
  std::vector<float> fc3Parameters = loadParametersFromFile("../parameters/fc3.txt");
  std::vector<float> fc4Parameters = loadParametersFromFile("../parameters/fc4.txt");
  std::vector<float> fc5Parameters = loadParametersFromFile("../parameters/fc5.txt");
  std::cout << "conv1 paramters: " << conv1Parameters.size() << std::endl;
  conv1->set_parameters(conv1Parameters);
  std::cout << "conv2 paramters: " << conv2Parameters.size() << std::endl;
  conv2->set_parameters(conv2Parameters);
  std::cout << "fc3 paramters: " << fc3Parameters.size() << std::endl;
  fc3->set_parameters(fc3Parameters);
  std::cout << "fc4 paramters: " << fc4Parameters.size() << std::endl;
  fc4->set_parameters(fc4Parameters);
  std::cout << "fc5 paramters: " << fc5Parameters.size() << std::endl;
  fc5->set_parameters(fc5Parameters);

  for (int i = 0; i < 5; i++) {
    std::cout << conv1Parameters[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << fc5Parameters[i] << " ";
  }
  std::cout << std::endl;

  // Test (Run forward)
  dnn.forward(dataset.test_data);
  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout << std::endl;
  std::cout << "Test acc: " << acc << std::endl;

  return 0;
}

