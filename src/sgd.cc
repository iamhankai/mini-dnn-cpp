#include "sgd.h"

void SGD::update(Vector::AlignedMapType& x, Vector::ConstAlignedMapType& dx) {
	x -= lr * dx;
}
