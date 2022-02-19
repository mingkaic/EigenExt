#include "unsupported/Eigen/CXX11/Tensor"

#include "EigenExt/sparse_util.h"
#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_SELECT_H
#define EIGEN_SPARSE_TENSOR_SELECT_H

namespace Eigen
{

template <typename BASE, typename LHS, typename RHS>
struct SparseTensorSelectOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	using FuncF = std::function<Scalar(const Scalar&,const Scalar&)>;

	inline SparseTensorSelectOp (const BASE& expr, const LHS& then, const RHS& other) :
		xpr_(expr), then_(then), other_(other) {}

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto xindices = xpr_.get_indices();

		auto ldata = then_.data();
		auto lindices = then_.get_indices();
		size_t ln = lindices.size();

		auto rdata = other_.data();
		auto rindices = other_.get_indices();
		size_t rn = rindices.size();

		size_t l = 0, r = 0;
		std::vector<Scalar> bufdata;
		std::vector<INDEX> bufidx;
		for (auto xindex : xindices)
		{
			// take elements from r for indices less than xindices[i]
			while (r < rn && rindices[r] < xindex)
			{
				// take rdata[r], rindices[r]
				bufdata.push_back(rdata[r]);
				bufidx.push_back(rindices[r]);
				++r;
			}
			while (l < ln && lindices[l] < xindex)
			{
				++l;
			}
			if (r < rn && l < ln) // ran out of all values to pick up
			{
				break;
			}
			if (lindices[l] == xindex)
			{
				// take ldata[l], lindices[l]
				bufdata.push_back(ldata[l]);
				bufidx.push_back(lindices[l]);
			}
		}
		block.set_nnz(bufdata.size());
		auto result = block.get_data();
		auto result_indices = block.get_indices();
		std::copy(bufdata.begin(), bufdata.end(), result);
		std::copy(bufidx.begin(), bufidx.end(), result_indices);
	}

private:
	const BASE& xpr_;
	const LHS& then_;
	const RHS& other_;
};

template <typename BASE, typename LHS, typename ROTHER>
struct SparseTensorSelectOp<BASE, LHS, ::Eigen::TensorMap<ROTHER>>
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	using FuncF = std::function<Scalar(const Scalar&,const Scalar&)>;

	inline SparseTensorSelectOp (const BASE& expr,
		const LHS& then,
		const ::Eigen::TensorMap<ROTHER>& other) :
		xpr_(expr), then_(then), other_(other) {}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		block = other_;
		auto xindices = xpr_.get_indices();

		auto ldata = then_.data();
		auto lindices = then_.get_indices();
		size_t ln = lindices.size();

		size_t lmatch = 0;
		auto result = block.data();
		for (auto xindex : xindices)
		{
			while (lmatch < ln && lindices[lmatch] < xindex)
			{
				++lmatch;
			}
			if (lmatch == ln) // ran out of left values to pick up
			{
				break;
			}
			if (lindices[lmatch] == xindex)
			{
				result[xindex] = ldata[lmatch];
			}
		}
	}

private:
	const BASE& xpr_;
	const LHS& then_;
	const ::Eigen::TensorMap<ROTHER>& other_;
};

template <typename BASE, typename LOTHER, typename RHS>
struct SparseTensorSelectOp<BASE, ::Eigen::TensorMap<LOTHER>, RHS>
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	using FuncF = std::function<Scalar(const Scalar&,const Scalar&)>;

	inline SparseTensorSelectOp (const BASE& expr,
		const ::Eigen::TensorMap<LOTHER>& then,
		const RHS& other) :
		xpr_(expr), then_(then), other_(other) {}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		block.setZero();
		auto xindices = xpr_.get_indices();

		auto ldata = then_.data();
		auto rdata = other_.data();
		auto rindices = other_.get_indices();
		size_t rn = rindices.size();

		auto result = block.data();
		for (size_t i = 0; i < rn; ++i)
		{
			result[rindices[i]] = rdata[i];
		}
		for (auto xindex : xindices)
		{
			result[xindex] = ldata[xindex];
		}
	}

private:
	const BASE& xpr_;
	const ::Eigen::TensorMap<LOTHER>& then_;
	const RHS& other_;
};

template <typename BASE, typename LOTHER, typename ROTHER>
struct SparseTensorSelectOp<BASE, ::Eigen::TensorMap<LOTHER>, ::Eigen::TensorMap<ROTHER>>
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	using FuncF = std::function<Scalar(const Scalar&,const Scalar&)>;

	inline SparseTensorSelectOp (const BASE& expr,
		const ::Eigen::TensorMap<LOTHER>& then,
		const ::Eigen::TensorMap<ROTHER>& other) :
		xpr_(expr), then_(then), other_(other) {}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		block = other_;
		auto xindices = xpr_.get_indices();

		auto result = block.data();
		auto data = then_.data();
		for (auto xidx : xindices)
		{
			result[xidx] = data[xidx];
		}
	}

private:
	const BASE& xpr_;
	const ::Eigen::TensorMap<LOTHER>& then_;
	const ::Eigen::TensorMap<ROTHER>& other_;
};

}

#endif // EIGEN_SPARSE_TENSOR_SELECT_H
