#ifndef MNIST_H
#define MNIST_H

#include <fstream>
#include <iostream>
#include <string>
#include "utils.h"

class MNIST {
private:
	std::string data_dir;

public:
	Matrix train_data;
	Matrix train_labels;
	Matrix test_data;
	Matrix test_labels;

	void read_mnist_data(std::string filename, Matrix& data);
	void read_mnist_label(std::string filename, Matrix& labels);	

	MNIST(std::string data_dir) : data_dir(data_dir) {}
	void read();
};

#endif /* MNIST_H */