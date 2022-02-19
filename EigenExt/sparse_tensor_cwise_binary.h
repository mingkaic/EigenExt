#include <list>
#include <unordered_set>

#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_CWISE_BINARY_H
#define EIGEN_SPARSE_TENSOR_CWISE_BINARY_H

namespace Eigen
{

template <typename LBASE, typename RBASE>
struct SparseTensorCwiseBinaryOp
{
	typedef typename LBASE::Scalar Scalar;
	typedef typename LBASE::Index Index;
	typedef typename LBASE::Dimensions Dimensions;
	static const size_t NumDims = LBASE::NumDims;

	using FuncF = std::function<Scalar(const Scalar&,const Scalar&)>;

	inline SparseTensorCwiseBinaryOp (
		const LBASE& lExpr, const RBASE& rExpr, const FuncF& binaryOp,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		lxpr_(lExpr), rxpr_(rExpr), binaryOp_(binaryOp), threshold_(threshold) {}

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto ldata = lxpr_.data();
		auto lxindices = lxpr_.get_indices();
		size_t ln = lxindices.size();

		auto rdata = rxpr_.data();
		auto rxindices = rxpr_.get_indices();
		size_t rn = rxindices.size();

		std::unordered_set<INDEX> idxset(
			lxindices.begin(), lxindices.end());
		idxset.insert(
			rxindices.begin(), rxindices.end());
		std::vector<INDEX> outindices(idxset.begin(), idxset.end());
		std::sort(outindices.begin(), outindices.end());

		block.allocate(outindices.size());
		auto result = block.get_data();
		auto rindices = block.get_indices();

		size_t nnz = 0;
		for (size_t i = 0, l = 0, r = 0, n = outindices.size(); i < n; ++i)
		{
			bool lmatch = l < ln && outindices[i] == lxindices[l];
			bool rmatch = r < rn && outindices[i] == rxindices[r];
			auto d = binaryOp_(
				lmatch ? ldata[l] : 0,
				rmatch ? rdata[r] : 0);
			if (not_close(d, threshold_))
			{
				rindices[nnz] = outindices[i];
				result[nnz++] = d;
			}
			if (lmatch)
			{
				++l;
			}
			if (rmatch)
			{
				++r;
			}
		}
		block.set_nnz(nnz);
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		auto ldata = lxpr_.data();
		auto lxindices = lxpr_.get_indices();
		size_t ln = lxindices.size();

		auto rdata = rxpr_.data();
		auto rxindices = rxpr_.get_indices();
		size_t rn = rxindices.size();

		auto result = block.data();
		size_t l = 0, r = 0;
		while (l < ln && r < rn)
		{
			if (lxindices[l] == rxindices[r])
			{
				result[lxindices[l]] = binaryOp_(ldata[l], rdata[r]);
				++l;
				++r;
			}
			else if (lxindices[l] < rxindices[r])
			{
				result[lxindices[l]] = binaryOp_(ldata[l], 0);
				++l;
			}
			else
			{
				result[rxindices[r]] = binaryOp_(0, rdata[r]);
				++r;
			}
		}
		while (l < ln)
		{
			result[lxindices[l]] = binaryOp_(ldata[l], 0);
			++l;
		}
		while (r < rn)
		{
			result[rxindices[r]] = binaryOp_(0, rdata[r]);
			++r;
		}
	}

private:
	const LBASE& lxpr_;
	const RBASE& rxpr_;
	const FuncF binaryOp_;
	const Scalar threshold_;
};

template <typename LBASE, typename ROTHER>
struct SparseTensorCwiseBinaryOp<LBASE,::Eigen::TensorMap<ROTHER>>
{
	typedef typename LBASE::Scalar Scalar;
	typedef typename LBASE::Index Index;
	typedef typename LBASE::Dimensions Dimensions;
	static const size_t NumDims = LBASE::NumDims;

	using FuncF = std::function<Scalar(const Scalar&,const Scalar&)>;

	inline SparseTensorCwiseBinaryOp (const LBASE& lExpr,
		const ::Eigen::TensorMap<ROTHER>& rExpr, const FuncF& binaryOp,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		lxpr_(lExpr), rxpr_(rExpr), binaryOp_(binaryOp)
	{
		auto rdims = rxpr_.dimensions();
		rn_ = internal::array_prod(rdims);
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		block = rxpr_;

		auto ldata = lxpr_.data();
		auto lxindices = lxpr_.get_indices();
		size_t ln = lxindices.size();

		auto result = block.data();
		for (size_t r = 0, l = 0; r < rn_; ++r)
		{
			auto& dst = result[r];
			dst = binaryOp_((l < ln && lxindices[l] == r) ? ldata[l++] : 0, dst);
		}
	}

private:
	const LBASE& lxpr_;
	const ::Eigen::TensorMap<ROTHER>& rxpr_;
	const FuncF binaryOp_;

	size_t rn_;
};

template <typename LOTHER, typename RBASE>
struct SparseTensorCwiseBinaryOp<::Eigen::TensorMap<LOTHER>,RBASE>
{
	typedef typename RBASE::Scalar Scalar;
	typedef typename RBASE::Index Index;
	typedef typename RBASE::Dimensions Dimensions;
	static const size_t NumDims = RBASE::NumDims;

	using FuncF = std::function<Scalar(const Scalar&,const Scalar&)>;

	inline SparseTensorCwiseBinaryOp (
		const ::Eigen::TensorMap<LOTHER>& lExpr,
		const RBASE& rExpr, const FuncF& binaryOp,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		lxpr_(lExpr), rxpr_(rExpr), binaryOp_(binaryOp)
	{
		auto ldims = lxpr_.dimensions();
		ln_ = internal::array_prod(ldims);
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		block = lxpr_;

		auto rdata = rxpr_.data();
		auto rxindices = rxpr_.get_indices();
		size_t rn = rxindices.size();

		auto result = block.data();
		for (size_t l = 0, r = 0; l < ln_; ++l)
		{
			auto& dst = result[l];
			dst = binaryOp_(dst, (r < rn && rxindices[r] == l) ? rdata[r++] : 0);
		}
	}

private:
	const ::Eigen::TensorMap<LOTHER>& lxpr_;
	const RBASE& rxpr_;
	const FuncF binaryOp_;

	size_t ln_;
};

}

#endif // EIGEN_SPARSE_TENSOR_CWISE_BINARY_H
