#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>
#include <random>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;

static std::default_random_engine generator;

// Normal distribution: N(mu, sigma^2)
inline void set_normal_random(float* arr, const int n, const float mu, 
												const float sigma) {

	std::normal_distribution<float> distribution(mu, sigma);
	for (int i = 0; i < n; i ++) {
		arr[i] = distribution(generator);
	}
}

// encode discrete values to one-hot values
inline Matrix one_hot_encode(const Matrix& y, const int n_value) {
	int n = y.cols();
	Matrix y_onehot = Matrix::Zero(n_value, n);
	for (int i = 0; i < n; i ++) {
		y_onehot((int)y(i), i) = 1;
	}
	return y_onehot;
}

#endif /* UTILS_H */