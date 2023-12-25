#include "Operator.h"

Matrix matrixMul(const Matrix &A, const Matrix &B, bool usingDevice) {
  if (usingDevice) {
    return A * B;
  } else {
    return A * B;
  }
}

void matrixColwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice) {
  if (usingDevice) {

  } else {
    des.colwise() += vec;
  }
}

void matrixRowwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice) {
  if (usingDevice) {

  } else {
    des.rowwise() += vec.transpose();
  }
}