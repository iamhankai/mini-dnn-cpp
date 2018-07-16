#include <iostream>
#include <Eigen/Dense>
#include "src/fully_connected.h"

//using Eigen::MatrixXd;

int main()
{
  Matrix m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;

  FullyConnected fc(2,2);
  SGD opt(0.01);
  fc.init();
  fc.forward(m);
  std::cout << fc.output() << std::endl;
  fc.backward(m, m);
  std::cout << fc.back_gradient() << std::endl;
  fc.update(opt);
}

