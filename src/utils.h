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
	for(int i = 0; i < n; i ++) {
		arr[i] = distribution(generator);
	}
}

#endif /* UTILS_H */