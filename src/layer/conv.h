#ifndef SRC_LAYER_CONV_H_
#define SRC_LAYER_CONV_H_

#include <vector>
#include "../layer.h"

class Conv: public Layer {
 private:
  const int dim_in;
  int dim_out;

  int channel_in;
  int height_in;
  int width_in;
  int channel_out;
  int height_kernel;
  int width_kernel;
  int stride;
  int pad_h;
  int pad_w;

  int height_out;
  int width_out;

  Matrix weight;  // weight param, size=channel_in*h_kernel*w_kernel*channel_out
  Vector bias;  // bias param, size = channel_out
  Matrix grad_weight;  // gradient w.r.t weight
  Vector grad_bias;  // gradient w.r.t bias

  std::vector<Matrix> data_cols;

  void init();

 public:
  Conv(int channel_in, int height_in, int width_in, int channel_out,
       int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
       int pad_h = 0) :
       dim_in(channel_in * height_in * width_in),
       channel_in(channel_in), height_in(height_in), width_in(width_in),
       channel_out(channel_out), height_kernel(height_kernel),
       width_kernel(width_kernel), stride(stride), pad_w(pad_w), pad_h(pad_h)
  { init(); }

  void forward(const Matrix& bottom);
  void backward(const Matrix& bottom, const Matrix& grad_top);
  void update(Optimizer& opt);
  void im2col(const Vector& image, Matrix& data_col);
  void col2im(const Matrix& data_col, Vector& image);
  int output_dim() { return dim_out; }
  std::vector<float> get_parameters() const;
  std::vector<float> get_derivatives() const;
  void set_parameters(const std::vector<float>& param);
};

#endif  // SRC_LAYER_CONV_H_
