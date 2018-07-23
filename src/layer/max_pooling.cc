#include "./max_pooling.h"
#include <math.h>
#include <limits>
#include <iostream>

void MaxPooling::init() {
  channel_out = channel_in;
  height_out = (1 + std::ceil((height_in - height_pool) * 1.0 / stride));
  width_out =   (1 + std::ceil((width_in - height_pool) * 1.0 / stride));
  dim_out = height_out * width_out * channel_out;
}

void MaxPooling::forward(const Matrix& bottom) {
  int n_sample = bottom.cols();
  int hw_in = height_in * width_in;
  int hw_pool = height_pool * width_pool;
  int hw_out = height_out * width_out;
  top.resize(dim_out, n_sample);
  top.setZero(); top.array() += std::numeric_limits<float>::lowest();
  max_idxs.resize(n_sample, std::vector<int>(dim_out, 0));
  for (int i = 0; i < n_sample; i ++) {
    Vector image = bottom.col(i);
    for (int c = 0; c < channel_in; c ++) {
      for (int i_out = 0; i_out < hw_out; i_out ++) {
        int step_h = i_out / width_out;
        int step_w = i_out % width_out;
        // left-top idx of window in raw image
        int start_idx = step_h * width_in * stride + step_w * stride;
        for (int i_pool = 0; i_pool < hw_pool; i_pool ++) {
          if (start_idx % width_in + i_pool % width_pool >= width_in ||
              start_idx / width_in + i_pool / width_pool >= height_in) {
            continue;  // out of range
          }
          int pick_idx = start_idx + (i_pool / width_pool) * width_in
                         + i_pool % width_pool + c * hw_in;
          if (image(pick_idx) >= top(c * hw_out + i_out, i)) {  // max pooling
            top(c * hw_out + i_out, i) = image(pick_idx);
            max_idxs[i][c * hw_out + i_out] = pick_idx;
          }
        }
      }
    }
  }
}

void MaxPooling::backward(const Matrix& bottom, const Matrix& grad_top) {
  grad_bottom.resize(bottom.rows(), bottom.cols());
  grad_bottom.setZero();
  for (int i = 0; i < max_idxs.size(); i ++) {  // i-th sample
    for (int j = 0; j < max_idxs[i].size(); j ++) {
      grad_bottom(max_idxs[i][j], i) += grad_top(j, i);
    }
  }
}
