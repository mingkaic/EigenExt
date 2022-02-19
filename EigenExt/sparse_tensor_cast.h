#include "EigenExt/sparse_util.h"
#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_CAST_H
#define EIGEN_SPARSE_TENSOR_CAST_H

namespace Eigen
{

template <typename BASE, typename T>
struct SparseTensorCastOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorCastOp (const BASE& expr,
		const T& threshold = std::numeric_limits<T>::epsilon()) :
		xpr_(expr), threshold_(threshold) {}

	template <typename INDEX>
	void write (iSparseTensorDst<T,INDEX>& block) const
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
			T d = data[i];
			if (not_close(d, threshold_))
			{
				rindices[nnz] = xindices[i];
				result[nnz++] = d;
			}
		}
		block.set_nnz(nnz);
	}

private:
	const BASE& xpr_;
	const T threshold_;
};

}

#endif // EIGEN_SPARSE_TENSOR_CAST_H
