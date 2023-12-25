#include "Operator.h"

Matrix matrixMul(const Matrix &A, const Matrix &B, bool usingDevice) {
  if (usingDevice) {
    Matrix result;
    result.resize(A.rows(), B.cols());
    dev_matrixMul(result.data(), A.data(), B.data(), A.rows(), A.cols(), B.cols());
    return result;
  } else {
    return A * B;
  }
}

void matrixColwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice) {
  if (usingDevice) {
    dev_matrixColwiseAddVec(des.data(), vec.data(), des.rows(), des.cols());
  } else {
    des.colwise() += vec;
  }
}

void matrixRowwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice) {
  if (usingDevice) {
    dev_matrixRowwiseAddVec(des.data(), vec.data(), des.rows(), des.cols());
  } else {
    des.rowwise() += vec.transpose();
  }
}