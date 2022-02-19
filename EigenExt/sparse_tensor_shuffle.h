#include "unsupported/Eigen/CXX11/Tensor"

#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_SHUFFLE_H
#define EIGEN_SPARSE_TENSOR_SHUFFLE_H

namespace Eigen
{

template <typename BASE, typename SHUFFLE>
struct SparseTensorShuffleOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorShuffleOp (const BASE& expr, const SHUFFLE& shuffle) :
		xpr_(expr), shuffle_(shuffle)
	{
		const Dimensions& input_dims = expr.dimensions();
		std::array<Index,NumDims> reverse_shuffle;
		is_identity_ = true;
		for (int i = 0; i < NumDims; ++i)
		{
			dimensions_[i] = input_dims[shuffle_[i]];
			reverse_shuffle[shuffle_[i]] = i;
			if (is_identity_ && shuffle_[i] != i)
			{
				is_identity_ = false;
			}
		}

		std::array<Index,NumDims> unshuffled_outstrides;
		unshuffled_outstrides[0] = 1;
		instrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			unshuffled_outstrides[i] = unshuffled_outstrides[i - 1] * dimensions_[i - 1];
			instrides_[i] = instrides_[i - 1] * input_dims[i - 1];
			fast_instrides_[i] = ::Eigen::internal::TensorIntDivisor<Index>(
				instrides_[i] > 0 ? instrides_[i] : Index(1));
		}

		for (int i = 0; i < NumDims; ++i)
		{
			outstrides_[i] = unshuffled_outstrides[reverse_shuffle[i]];
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
		for (int i = NumDims - 1; i > 0; --i)
		{
			const Index idx = index / fast_instrides_[i];
			output_index += idx * outstrides_[i];
			index -= idx * instrides_[i];
		}
		return output_index + index * outstrides_[0];
	}

	const BASE& xpr_;
	const SHUFFLE shuffle_;

	bool is_identity_;
	Dimensions dimensions_;
	std::array<Index, NumDims> instrides_;
	std::array<Index, NumDims> outstrides_;
	std::array<::Eigen::internal::TensorIntDivisor<Index>,NumDims> fast_instrides_;
};

}

#endif // EIGEN_SPARSE_TENSOR_SHUFFLE_H
