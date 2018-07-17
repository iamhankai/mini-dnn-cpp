#include "network.h"

void Network::forward(const Matrix& input) {
	if(layers.empty())
		return;
	layers[0]->forward(input);
	for (int i = 1; i < layers.size(); i ++) {
		layers[i]->forward(layers[i-1]->output());
	}
}

void Network::backward(const Matrix& input, const Matrix& target) {
	int n_layer = layers.size();
	// 0 layer
	if (n_layer <= 0)
		return;
	// 1 layer
	loss->evaluate(layers[n_layer-1]->output(), target);
	if (n_layer == 1) {
		layers[0]->backward(input, loss->back_gradient());
		return;
	}
	// >1 layers
	layers[n_layer-1]->backward(layers[n_layer-2]->output(), loss->back_gradient());
	for (int i = n_layer-2; i > 0; i --) {
		layers[i]->backward(layers[i-1]->output(), layers[i+1]->back_gradient());
	}
	layers[0]->backward(input, layers[1]->back_gradient());
}

void Network::update(SGD& opt) {
	for (int i = 0; i < layers.size(); i ++) {
		layers[i]->update(opt);
	}
}