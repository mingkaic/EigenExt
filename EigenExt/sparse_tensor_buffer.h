#include <algorithm>
#include <unordered_map>

#include "unsupported/Eigen/CXX11/Tensor"

#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_BUFFER_H
#define EIGEN_SPARSE_TENSOR_BUFFER_H

namespace Eigen
{

template <typename T, typename INDEX>
struct AccumulateBuffer final
{
	AccumulateBuffer (const T& threshold = std::numeric_limits<T>::epsilon()) :
		threshold_(threshold) {}

	template <typename OTHER>
	void assign_nonzeros (const ::Eigen::TensorMap<OTHER>& other,
		const std::function<T(const T&,const T&)>& f)
	{
		auto data = other.data();
		for (auto& i2v : idx2val_)
		{
			i2v.second = f(i2v.second, data[i2v.first]);
		}
	}

	void unaryExpr (const std::function<T(const T&)>& f)
	{
		for (auto& i2v : idx2val_)
		{
			i2v.second = f(i2v.second);
		}
	}

	void write (iSparseTensorDst<T,INDEX>& block) const
	{
		block.allocate(idx2val_.size());
		auto result = block.get_data();
		auto rindices = block.get_indices();

		size_t nnz = 0;
		for (auto& i2v : idx2val_)
		{
			if (not_close(i2v.second, threshold_))
			{
				rindices[nnz++] = i2v.first;
			}
		}
		std::sort(rindices, rindices + nnz);
		for (size_t i = 0; i < nnz; ++i)
		{
			auto ridx = rindices[i];
			result[i] = idx2val_.at(ridx);
		}
		block.set_nnz(nnz);
	}

	std::unordered_map<INDEX,T> idx2val_;

	const T threshold_;
};

}

#endif // EIGEN_SPARSE_TENSOR_BUFFER_H
