#include "./mnist.h"

int ReverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNIST::read_mnist_data(std::string filename, Matrix& data) {
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    unsigned char label;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    n_rows = ReverseInt(n_rows);
    n_cols = ReverseInt(n_cols);

    data.resize(n_cols * n_rows, number_of_images);
    for (int i = 0; i < number_of_images; i++) {
      for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
          unsigned char image = 0;
          file.read((char*)&image, sizeof(image));
          data(r * n_cols + c, i) = (float)image;
        }
      }
    }
  }
}

void MNIST::read_mnist_label(std::string filename, Matrix& labels) {
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);

    labels.resize(1, number_of_images);
    for (int i = 0; i < number_of_images; i++) {
      unsigned char label = 0;
      file.read((char*)&label, sizeof(label));
      labels(0, i) = (float)label;
    }
  }
}

void MNIST::read() {
  read_mnist_data(data_dir + "train-images-idx3-ubyte", train_data);
  read_mnist_data(data_dir + "t10k-images-idx3-ubyte", test_data);
  read_mnist_label(data_dir + "train-labels-idx1-ubyte", train_labels);
  read_mnist_label(data_dir + "t10k-labels-idx1-ubyte", test_labels);
}
