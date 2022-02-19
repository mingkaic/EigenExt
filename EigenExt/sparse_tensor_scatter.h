#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_SCATTER_H
#define EIGEN_SPARSE_TENSOR_SCATTER_H

namespace Eigen
{

template <typename BASE, typename DIM>
struct SparseTensorScatterOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorScatterOp (const BASE& expr,
		const std::array<DIM,NumDims>& strides, const Dimensions& dimensions) :
		xpr_(expr), strides_(strides), dimensions_(dimensions)
	{
		auto indims = expr.dimensions();
		is_identity_ = true;
		for (int i = 0; i < NumDims; ++i)
		{
			is_identity_ = is_identity_ && strides[i] == 1 && dimensions[i] == indims[i];
		}
		instrides_[0] = 1;
		outstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			instrides_[i] = instrides_[i - 1] * indims[i - 1];
			outstrides_[i] = outstrides_[i - 1] * dimensions_[i - 1];
		}
	}

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();
		block.set_nnz(n);
		auto result = block.get_data();
		auto rindices = block.get_indices();

		std::copy(data, data + n, result);
		if (is_identity_)
		{
			std::copy(xindices.begin(), xindices.end(), rindices);
			return;
		}
		for (size_t i = 0; i < n; ++i)
		{
			rindices[i] = dst_index(xindices[i]);
		}
	}

private:
	inline Index dst_index (Index index) const
	{
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index idx = index / instrides_[i];
			output_index += idx * strides_[i] * outstrides_[i];
			index -= idx * instrides_[i];
		}
		return output_index;
	}

	const BASE& xpr_;
	const std::array<DIM,NumDims> strides_;
	const Dimensions dimensions_;
	bool is_identity_;

	std::array<Index,NumDims> instrides_;
	std::array<Index,NumDims> outstrides_;
};

}

#endif // EIGEN_SPARSE_TENSOR_SCATTER_H
