#ifndef _Operator_H_
#define _Operator_H_

#include <Eigen/Core>
#include "../utils.h"

Matrix matrixMul(const Matrix &A, const Matrix &B, bool usingDevice=false);
void matrixColwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice=false);
void matrixRowwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice=false);

#endif