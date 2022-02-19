#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_STRIDE_H
#define EIGEN_SPARSE_TENSOR_STRIDE_H

namespace Eigen
{

template <typename BASE, typename DIM>
struct SparseTensorStrideOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorStrideOp (const BASE& expr,
		const std::array<DIM,NumDims>& strides) :
		xpr_(expr), strides_(strides)
	{
		auto indims = expr.dimensions();
		is_identity_ = true;
		for (int i = 0; i < NumDims; ++i)
		{
			dimensions_[i] = indims[i] / strides[i] +
				((indims[i] % strides[i]) == 0 ? 0 : 1);
			is_identity_ = is_identity_ && strides[i] == 1;
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

		if (is_identity_)
		{
			block.set_nnz(n);
			auto result = block.get_data();
			auto rindices = block.get_indices();
			std::copy(data, data + n, result);
			std::copy(xindices.begin(), xindices.end(), rindices);
			return;
		}

		size_t nout = internal::array_prod(dimensions_);
		std::vector<Scalar> tmpdata;
		std::vector<size_t> tmpindx;
		tmpdata.reserve(nout);
		tmpindx.reserve(nout);
		for (size_t i = 0; i < n; ++i)
		{
			auto ridx = dst_index(xindices[i], nout);
			if (ridx < nout)
			{
				tmpdata.push_back(data[i]);
				tmpindx.push_back(ridx);
			}
		}
		block.set_nnz(tmpdata.size());
		auto result = block.get_data();
		auto rindices = block.get_indices();
		std::copy(tmpdata.begin(), tmpdata.end(), result);
		std::copy(tmpindx.begin(), tmpindx.end(), rindices);
	}

private:
	inline Index dst_index (Index index, Index sentinel) const
	{
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index idx = index / instrides_[i];
			if (idx % strides_[i] != 0)
			{
				return sentinel;
			}
			output_index += (idx / strides_[i]) * outstrides_[i];
			index -= idx * instrides_[i];
		}
		return output_index;
	}

	const BASE& xpr_;
	const std::array<DIM,NumDims> strides_;

	bool is_identity_;
	Dimensions dimensions_;
	std::array<Index,NumDims> instrides_;
	std::array<Index,NumDims> outstrides_;
};

}

#endif // EIGEN_SPARSE_TENSOR_STRIDE_H
