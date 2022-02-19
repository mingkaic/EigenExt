#include "unsupported/Eigen/CXX11/Tensor"

#include "EigenExt/sparse_shape.h"
#include "EigenExt/sparse_tensor_assign.h"
#include "EigenExt/sparse_tensor_broadcast.h"
#include "EigenExt/sparse_tensor_cast.h"
#include "EigenExt/sparse_tensor_chip.h"
#include "EigenExt/sparse_tensor_contract.h"
#include "EigenExt/sparse_tensor_convolve.h"
#include "EigenExt/sparse_tensor_cwise_unary.h"
#include "EigenExt/sparse_tensor_cwise_binary.h"
#include "EigenExt/sparse_tensor_reduce.h"
#include "EigenExt/sparse_tensor_reverse.h"
#include "EigenExt/sparse_tensor_select.h"
#include "EigenExt/sparse_tensor_shuffle.h"
#include "EigenExt/sparse_tensor_slice.h"
#include "EigenExt/sparse_tensor_scatter.h"
#include "EigenExt/sparse_tensor_stride.h"
#include "EigenExt/sparse_tensor_pad.h"

#ifndef EIGEN_SPARSE_TENSOR_SRC_H
#define EIGEN_SPARSE_TENSOR_SRC_H

namespace Eigen
{

template<typename T, size_t RANK, typename IDX=size_t>
struct SparseTensorSrc final
{
	typedef SparseTensorSrc<T,RANK,IDX> Self;
	typedef T Scalar;
	typedef DimT Index;
	typedef ShapeT<RANK> Dimensions;
	static const size_t NumDims = RANK;

	inline SparseTensorSrc (const Dimensions& dimensions,
		size_t nnz, const IDX* indices, const T* value) :
		shape_(dimensions), data_(value), indices_(indices, indices + nnz) {}

	inline SparseTensorSrc (const Dimensions& dimensions, iSparseTensorDst<T,IDX>& block) :
		shape_(dimensions), data_(block.get_data()),
		indices_(block.get_indices(), block.get_indices() + block.non_zeros()) {}

	inline size_t non_zeros (void) const { return indices_.size(); }

	inline std::vector<size_t> get_indices (void) const
	{
		return std::vector<size_t>(indices_.begin(), indices_.end());
	}

	inline const Dimensions& dimensions (void) const { return shape_; }

	inline const Scalar* data (void) const { return data_; }

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto src = data();
		auto xindices = get_indices();
		size_t n = xindices.size();

		block.set_nnz(n);
		auto result = block.get_data();
		auto rindices = block.get_indices();
		std::copy(src, src + n, result);
		std::copy(xindices.begin(), xindices.end(), rindices);
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		auto src = data();
		auto xindices = get_indices();

		auto result = block.data();
		for (size_t i = 0, n = xindices.size(); i < n; ++i)
		{
			result[xindices[i]] = src[i];
		}
	}

	template <typename OTHER>
	void overwrite (::Eigen::TensorMap<OTHER>& block) const
	{
		auto src = data();
		auto xindices = get_indices();

		size_t x = 0;
		size_t xn = xindices.size();
		auto result = block.data();
		for (size_t i = 0, n = internal::array_prod(block.dimensions()); i < n; ++i)
		{
			if (x < xn && i == xindices[x])
			{
				result[i] = src[x];
				++x;
			}
			else
			{
				result[i] = 0;
			}
		}
	}

	template <typename DIM, typename REDUCE_OP>
	inline SparseTensorReduceOp<Self,DIM,REDUCE_OP> reduce (const std::set<DIM>& dims) const
	{
		return SparseTensorReduceOp<Self,DIM,REDUCE_OP>(*this, dims);
	}

	template <typename SHUFFLE>
	inline SparseTensorShuffleOp<Self,SHUFFLE> shuffle (const SHUFFLE& shuffle) const
	{
		return SparseTensorShuffleOp<Self,SHUFFLE>(*this, shuffle);
	}

	template <typename DIM>
	inline SparseTensorBroadcastOp<Self,DIM> broadcast (
		const std::array<DIM,NumDims>& bcast) const
	{
		return SparseTensorBroadcastOp<Self,DIM>(*this, bcast);
	}

	template <typename DIM>
	inline SparseTensorChipOp<Self,DIM> chip (const DIM& i, const DIM& axis,
		const Dimensions& dimensions) const
	{
		return SparseTensorChipOp<Self,DIM>(*this, i, axis, dimensions);
	}

	template <typename DIM>
	inline SparseTensorSliceOp<Self,DIM> slice (
		const std::array<DIM,NumDims>& offsets,
		const std::array<DIM,NumDims>& extents) const
	{
		return SparseTensorSliceOp<Self,DIM>(*this, offsets, extents);
	}

	template <typename DIM>
	inline SparseTensorPadOp<Self,DIM> pad (
		const std::array<std::pair<DIM,DIM>,NumDims>& paddings) const
	{
		return SparseTensorPadOp<Self,DIM>(*this, paddings);
	}

	template <typename DIM>
	inline SparseTensorScatterOp<Self,DIM> scatter (
		const std::array<DIM,NumDims>& strides, const Dimensions& dimensions) const
	{
		return SparseTensorScatterOp<Self,DIM>(*this, strides, dimensions);
	}

	template <typename DIM>
	inline SparseTensorStrideOp<Self,DIM> stride (
		const std::array<DIM,NumDims>& strides) const
	{
		return SparseTensorStrideOp<Self,DIM>(*this, strides);
	}

	inline SparseTensorReverseOp<Self> reverse (const std::array<bool,NumDims>& rev) const
	{
		return SparseTensorReverseOp<Self>(*this, rev);
	}

	inline SparseTensorCwiseUnaryOp<Self> cwiseUnary (
		const std::function<T(const T&)>& f,
		const T& threshold = std::numeric_limits<T>::epsilon()) const
	{
		return SparseTensorCwiseUnaryOp<Self>(*this, f, threshold);
	}

	template <typename RHS>
	inline SparseTensorCwiseBinaryOp<Self,RHS> cwiseBinary (const RHS& other,
		const std::function<T(const T&,const T&)>& f,
		const T& threshold = std::numeric_limits<T>::epsilon()) const
	{
		return SparseTensorCwiseBinaryOp<Self,RHS>(*this, other, f, threshold);
	}

	template <typename LHS>
	inline SparseTensorCwiseBinaryOp<LHS,Self> cwiseRBinary (const LHS& other,
		const std::function<T(const T&,const T&)>& f,
		const T& threshold = std::numeric_limits<T>::epsilon()) const
	{
		return SparseTensorCwiseBinaryOp<LHS,Self>(other, *this, f, threshold);
	}

	template <typename RHS, typename CIDX>
	inline SparseTensorContractOp<Self,RHS,CIDX> contract (const RHS& other,
		const ContractDimsT<CIDX>& dims) const
	{
		return SparseTensorContractOp<Self,RHS,CIDX>(*this, other, dims);
	}

	template <typename LHS, typename CIDX>
	inline SparseTensorContractOp<LHS,Self,CIDX> rContract (const LHS& other,
		const ContractDimsT<CIDX>& dims) const
	{
		return SparseTensorContractOp<LHS,Self,CIDX>(other, *this, dims);
	}

	template <typename RHS, typename DIM>
	inline SparseTensorConvolveOp<Self,RHS,DIM> convolve (const RHS& other,
		const std::array<DIM,NumDims>& dims) const
	{
		return SparseTensorConvolveOp<Self,RHS,DIM>(*this, other, dims);
	}

	template <typename LHS, typename DIM>
	inline SparseTensorConvolveOp<LHS,Self,DIM> rConvolve (const LHS& other,
		const std::array<DIM,NumDims>& dims) const
	{
		return SparseTensorConvolveOp<LHS,Self,DIM>(other, *this, dims);
	}

	inline SparseTensorAssignOp<Self> assign (
		const std::function<T(const T&,const T&)>& f =
		std::function<T(const T&,const T&)>()) const
	{
		return SparseTensorAssignOp<Self>(*this, f);
	}

	inline SparseTensorAssignIndexOp<Self> assignIndex (
		const std::function<void(T&,const T&,size_t)>& f) const
	{
		return SparseTensorAssignIndexOp<Self>(*this, f);
	}

	template <typename LHS, typename RHS>
	inline SparseTensorSelectOp<Self,LHS,RHS> select (const LHS& then, const RHS& otherwise) const
	{
		return SparseTensorSelectOp<Self,LHS,RHS>(*this, then, otherwise);
	}

	template <typename OTHER_TYPE>
	inline SparseTensorCastOp<Self,OTHER_TYPE> cast (void) const
	{
		return SparseTensorCastOp<Self,OTHER_TYPE>(*this);
	}

private:
	Dimensions shape_;
	const Scalar* data_;
	const std::vector<IDX> indices_;
};

}

#endif // EIGEN_SPARSE_TENSOR_SRC_H
