#include <array>

#ifndef EIGEN_SPARSE_TENSOR_SHAPE_H
#define EIGEN_SPARSE_TENSOR_SHAPE_H

namespace Eigen
{

using DimT = size_t;

template <size_t RANK>
using ShapeT = std::array<DimT,RANK>;

}

#endif // EIGEN_SPARSE_TENSOR_SHAPE_H
