#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_BROADCAST_H
#define EIGEN_SPARSE_TENSOR_BROADCAST_H

namespace Eigen
{

template <typename BASE, typename DIM>
struct SparseTensorBroadcastOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorBroadcastOp (const BASE& expr,
		const std::array<DIM,NumDims>& bcast) :
		xpr_(expr), bcast_(bcast), indims_(expr.dimensions())
	{
		instrides_[0] = 1;
		outstrides_[0] = 1;
		bcast_strides_[0] = 1;
		dimensions_[0] = indims_[0] * bcast_[0];
		for (int i = 1; i < NumDims; ++i)
		{
			dimensions_[i] = indims_[i] * bcast_[i];
			instrides_[i] = instrides_[i - 1] * indims_[i - 1];
			outstrides_[i] = outstrides_[i - 1] * dimensions_[i - 1];
			bcast_strides_[i] = bcast_strides_[i - 1] * bcast[i - 1];
		}
	}

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();
		auto multiples = internal::array_prod(bcast_);
		auto nout = n * multiples;

		block.set_nnz(nout);
		auto result = block.get_data();
		auto rindices = block.get_indices();

		if (multiples == 1)
		{
			std::copy(data, data + n, result);
			std::copy(xindices.begin(), xindices.end(), rindices);
			return;
		}

		std::vector<INDEX> indices;
		indices.reserve(nout);
		for (INDEX xidx : xindices)
		{
			for (size_t i = 0; i < multiples; ++i)
			{
				indices.push_back(dst_index(xidx, i));
			}
		}
		std::vector<size_t> sindices(nout);
		std::iota(sindices.begin(), sindices.end(), 0);
		std::sort(sindices.begin(), sindices.end(),
		[&indices](const size_t& l, const size_t& r)
		{
			return indices[l] < indices[r];
		});

		for (size_t i = 0; i < nout; ++i)
		{
			size_t sidx = sindices[i];
			size_t didx = sidx / multiples;
			result[i] = data[didx];
			rindices[i] = indices[sidx];
		}
	}

private:
	inline Index dst_index (Index index, Index bcast_idx) const
	{
		Index output_index = 0;
		for (int i = NumDims - 1; i > 0; --i)
		{
			const Index iidx = index / instrides_[i];
			const Index bidx = bcast_idx / bcast_strides_[i];
			output_index += iidx * outstrides_[i] +
				indims_[i] * bidx * instrides_[i] * bcast_strides_[i];
			index -= iidx * instrides_[i];
			bcast_idx -= bidx * bcast_strides_[i];
		}
		return output_index + index * outstrides_[0] +
			indims_[0] * bcast_idx * instrides_[0] * bcast_strides_[0];
	}

	const BASE& xpr_;
	const std::array<DIM,NumDims> bcast_;
	const Dimensions indims_;

	Dimensions dimensions_;
	std::array<Index, NumDims> bcast_strides_;
	std::array<Index, NumDims> instrides_;
	std::array<Index, NumDims> outstrides_;
};

}

#endif // EIGEN_SPARSE_TENSOR_BROADCAST_H
