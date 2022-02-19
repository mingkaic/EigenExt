#include "EigenExt/sparse_util.h"
#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_CWISE_UNARY_H
#define EIGEN_SPARSE_TENSOR_CWISE_UNARY_H

namespace Eigen
{

template <typename BASE>
struct SparseTensorCwiseUnaryOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	using FuncF = std::function<Scalar(const Scalar&)>;

	inline SparseTensorCwiseUnaryOp (const BASE& expr, const FuncF& unaryOp,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		xpr_(expr), unaryOp_(unaryOp), threshold_(threshold) {}

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();

		block.allocate(n);
		auto result = block.get_data();
		auto rindices = block.get_indices();

		size_t nnz = 0;
		for (size_t i = 0; i < n; ++i)
		{
			auto d = unaryOp_(data[i]);
			if (not_close(d, threshold_))
			{
				rindices[nnz] = xindices[i];
				result[nnz++] = d;
			}
		}
		block.set_nnz(nnz);
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();

		auto result = block.data();
		for (size_t i = 0; i < n; ++i)
		{
			result[xindices[i]] = unaryOp_(data[i]);
		}
	}

	template <typename OTHER>
	void negative_write (::Eigen::TensorMap<OTHER>& block) const
	{
		auto xindices = xpr_.get_indices();
		size_t xn = xindices.size();

		auto result = block.data();
		auto n = internal::array_prod(block.dimensions());
		size_t x = 0;
		for (size_t i = 0; i < n; ++i)
		{
			if (x < xn && i == xindices[x])
			{
				++x;
			}
			else
			{
				result[i] = unaryOp_(0);
			}
		}
	}

private:
	const BASE& xpr_;
	const FuncF unaryOp_;
	const Scalar threshold_;
};

}

#endif // EIGEN_SPARSE_TENSOR_CWISE_UNARY_H
