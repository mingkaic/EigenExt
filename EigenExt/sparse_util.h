#include <cstdlib>
#include <type_traits>

#ifndef EIGEN_SPARSE_TENSOR_UTIL_H
#define EIGEN_SPARSE_TENSOR_UTIL_H

namespace Eigen
{

template <typename T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
inline bool not_close (const T& val, const T& thresh)
{
	return val > thresh;
}

template <typename T, typename std::enable_if<!std::is_unsigned<T>::value>::type* = nullptr>
inline bool not_close (const T& val, const T& thresh)
{
	return std::abs(val) > thresh;
}

}

#endif // EIGEN_SPARSE_TENSOR_UTIL_H
