#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_SLICE_H
#define EIGEN_SPARSE_TENSOR_SLICE_H

namespace Eigen
{

template <typename BASE, typename DIM>
struct SparseTensorSliceOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorSliceOp (const BASE& expr,
		const std::array<DIM,NumDims>& offsets,
		const std::array<DIM,NumDims>& extents) :
		xpr_(expr), offsets_(offsets), indims_(expr.dimensions())
	{
		std::copy(extents.begin(), extents.end(), dimensions_.begin());
		instrides_[0] = 1;
		outstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			instrides_[i] = instrides_[i - 1] * indims_[i - 1];
			outstrides_[i] = outstrides_[i - 1] * dimensions_[i - 1];
		}
		size_t nin = internal::array_prod(indims_);
		size_t nout = internal::array_prod(dimensions_);
		assert(nin >= nout);
		is_identity_ = nin == nout;
	}

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();
		size_t nout = internal::array_prod(dimensions_);

		if (is_identity_)
		{
			block.set_nnz(n);
			auto result = block.get_data();
			auto rindices = block.get_indices();
			std::copy(data, data + n, result);
			std::copy(xindices.begin(), xindices.end(), rindices);
			return;
		}

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
			if (idx < offsets_[i] || (idx - offsets_[i]) >= dimensions_[i])
			{
				return sentinel;
			}
			output_index += (idx - offsets_[i]) * outstrides_[i];
			index -= idx * instrides_[i];
		}
		return output_index;
	}

	const BASE& xpr_;
	const std::array<DIM,NumDims> offsets_;
	const Dimensions indims_;

	Dimensions dimensions_;
	std::array<Index,NumDims> instrides_;
	std::array<Index,NumDims> outstrides_;
	bool is_identity_;
};

}

#endif // EIGEN_SPARSE_TENSOR_SLICE_H
