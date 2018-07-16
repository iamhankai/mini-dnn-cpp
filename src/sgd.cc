#include "sgd.h"

void SGD::update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw) {
	w -= lr * (dw + decay * w);
}
