#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_CHIP_H
#define EIGEN_SPARSE_TENSOR_CHIP_H

namespace Eigen
{

template <typename BASE, typename DIM>
struct SparseTensorChipOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorChipOp (const BASE& expr, DIM start_idx, DIM dim,
		const Dimensions& dimensions,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		xpr_(expr), dimensions_(dimensions), start_idx_(start_idx),
		concat_dim_(dim), threshold_(threshold)
	{
		auto indims = expr.dimensions();
		instrides_[0] = 1;
		outstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			instrides_[i] = instrides_[i - 1] * indims[i - 1];
			outstrides_[i] = outstrides_[i - 1] * dimensions_[i - 1];
		}
	}

	template <typename INDEX>
	void fast_write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		assert(block.alloc_size() > 0);
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();

		auto result = block.get_data();
		auto rindices = block.get_indices();
		size_t m = block.non_zeros();
		for (size_t i = 0; i < n; ++i)
		{
			if (not_close(data[i], threshold_))
			{
				auto dstidx = dst_index(xindices[i]);
				result[m] = data[i];
				rindices[m++] = dstidx;
			}
		}
		block.set_nnz(m);
	}

	template <typename INDEX>
	void fast_write_finalize (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		fast_write(block);
		auto result = block.get_data();
		auto rindices = block.get_indices();
		auto n = block.non_zeros();

		std::vector<Scalar> buf_data(result, result + n);
		std::vector<INDEX> buf_indices(rindices, rindices + n);

		std::vector<size_t> sindices(n);
		std::iota(sindices.begin(), sindices.end(), 0);
		std::sort(sindices.begin(), sindices.end(),
		[&buf_indices](const size_t& l, const size_t& r)
		{
			return buf_indices[l] < buf_indices[r];
		});
		for (size_t i = 0; i < n; ++i)
		{
			auto sidx = sindices[i];
			result[i] = buf_data[sidx];
			rindices[i] = buf_indices[sidx];
		}
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();

		auto dst_data = block.data();
		for (size_t i = 0; i < n; ++i)
		{
			dst_data[dst_index(xindices[i])] = data[i];
		}
	}

private:
	inline Index dst_index (Index index) const
	{
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			Index idx = index / instrides_[i];
			if (concat_dim_ == i)
			{
				idx += start_idx_;
			}
			output_index += idx * outstrides_[i];
			index -= idx * instrides_[i];
		}
		return output_index;
	}

	const BASE& xpr_;
	const Dimensions dimensions_;
	const Index start_idx_;
	const Index concat_dim_;
	const Scalar threshold_;

	std::array<Index,NumDims> instrides_;
	std::array<Index,NumDims> outstrides_;
};

}

#endif // EIGEN_SPARSE_TENSOR_CHIP_H
