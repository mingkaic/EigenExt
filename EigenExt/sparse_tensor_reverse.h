#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_REVERSE_H
#define EIGEN_SPARSE_TENSOR_REVERSE_H

namespace Eigen
{

template <typename BASE>
struct SparseTensorReverseOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorReverseOp (const BASE& expr, const std::array<bool,NumDims>& rev) :
		xpr_(expr), dimensions_(expr.dimensions()), rev_(rev)
	{
		is_identity_ = true;
		for (int i = 0; i < NumDims; ++i)
		{
			is_identity_ = is_identity_ && (!rev_[i] || dimensions_[i] == 1);
		}
		strides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			strides_[i] = strides_[i - 1] * dimensions_[i - 1];
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

		if (is_identity_)
		{
			std::copy(data, data + n, result);
			std::copy(xindices.begin(), xindices.end(), rindices);
			return;
		}

		std::vector<INDEX> indices;
		indices.reserve(n);
		for (INDEX xidx : xindices)
		{
			indices.push_back(dst_index(xidx));
		}
		std::vector<size_t> sindices(n);
		std::iota(sindices.begin(), sindices.end(), 0);
		std::sort(sindices.begin(), sindices.end(),
		[&indices](const size_t& l, const size_t& r)
		{
			return indices[l] < indices[r];
		});

		for (size_t i = 0; i < n; ++i)
		{
			size_t sidx = sindices[i];
			result[i] = data[sidx];
			rindices[i] = indices[sidx];
		}
	}

private:
	inline Index dst_index (Index index) const
	{
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index idx = index / strides_[i];
			if (rev_[i])
			{
				output_index += (dimensions_[i] - idx - 1) * strides_[i];
			}
			else
			{
				output_index += idx * strides_[i];
			}
			index -= idx * strides_[i];
		}
		return output_index;
	}

	const BASE& xpr_;
	const Dimensions dimensions_;
	const std::array<bool,NumDims> rev_;
	bool is_identity_;

	std::array<Index,NumDims> strides_;
};

}

#endif // EIGEN_SPARSE_TENSOR_REVERSE_H
