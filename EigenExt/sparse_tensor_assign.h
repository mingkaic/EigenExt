#include <unordered_set>

#include "EigenExt/sparse_util.h"
#include "EigenExt/sparse_tensor_buffer.h"

#ifndef EIGEN_SPARSE_TENSOR_ASSIGN_H
#define EIGEN_SPARSE_TENSOR_ASSIGN_H

namespace Eigen
{

template <typename BASE>
struct SparseTensorAssignOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	using FuncF = std::function<Scalar(const Scalar&,const Scalar&)>;

	inline SparseTensorAssignOp (const BASE& expr,
		const FuncF& binaryOp = FuncF()) :
		xpr_(expr), binaryOp_(binaryOp) {}

	template <typename INDEX>
	void write (AccumulateBuffer<Scalar,INDEX>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();

		auto& idx2val = block.idx2val_;
		if (binaryOp_)
		{
			for (size_t i = 0; i < n; ++i)
			{
				auto match = idx2val.find(xindices[i]);
				idx2val[xindices[i]] = binaryOp_(
					match == idx2val.end() ? 0 : match->second, data[i]);
			}
		}
		else
		{
			for (size_t i = 0; i < n; ++i)
			{
				idx2val[xindices[i]] = data[i];
			}
		}
	}

	template <typename INDEX>
	void overwrite (AccumulateBuffer<Scalar,INDEX>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();

		auto& idx2val = block.idx2val_;
		if (binaryOp_)
		{
			std::unordered_set<INDEX> xkeys;
			for (auto& i2v : idx2val)
			{
				xkeys.emplace(i2v.first);
			}
			for (size_t i = 0; i < n; ++i)
			{
				auto match = idx2val.find(xindices[i]);
				if (match == idx2val.end())
				{
					idx2val[xindices[i]] = binaryOp_(0, data[i]);
				}
				else
				{
					idx2val[xindices[i]] = binaryOp_(match->second, data[i]);
					xkeys.erase(match->first);
				}
			}
			for (auto& xkey : xkeys)
			{
				auto& dst = idx2val[xkey];
				dst = binaryOp_(dst, 0);
			}
		}
		else
		{
			idx2val.clear();
			for (size_t i = 0; i < n; ++i)
			{
				idx2val[xindices[i]] = data[i];
			}
		}
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();

		auto result = block.data();

		if (binaryOp_)
		{
			for (size_t i = 0; i < n; ++i)
			{
				auto& dst = result[xindices[i]];
				dst = binaryOp_(dst, data[i]);
			}
		}
		else
		{
			for (size_t i = 0; i < n; ++i)
			{
				result[xindices[i]] = data[i];
			}
		}
	}

	template <typename OTHER>
	void overwrite (::Eigen::TensorMap<OTHER>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t xn = xindices.size();

		auto result = block.data();
		auto n = internal::array_prod(block.dimensions());
		auto op = binaryOp_ ? binaryOp_ : [](Scalar l, Scalar r) { return r; };

		size_t x = 0;
		for (size_t i = 0; i < n; ++i)
		{
			if (x < xn && i == xindices[x])
			{
				result[i] = op(result[i], data[x]);
				++x;
			}
			else
			{
				result[i] = op(result[i], 0);
			}
		}
	}

private:
	const BASE& xpr_;
	const FuncF binaryOp_;
};

template <typename BASE>
struct SparseTensorAssignIndexOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	using FuncF = std::function<void(Scalar&,const Scalar&,size_t)>;

	inline SparseTensorAssignIndexOp (const BASE& expr, const FuncF& assignOp) :
		xpr_(expr), assignOp_(assignOp) {}

	template <typename INDEX>
	void write (AccumulateBuffer<Scalar,INDEX>& block) const
	{
		auto data = xpr_.data();
		auto xindices = xpr_.get_indices();
		size_t n = xindices.size();

		auto& idx2val = block.idx2val_;
		for (size_t i = 0; i < n; ++i)
		{
			if (idx2val.find(xindices[i]) == idx2val.end())
			{
				idx2val.emplace(xindices[i], 0);
			}
			assignOp_(idx2val[xindices[i]], data[i], xindices[i]);
		}
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
			auto& dst = result[xindices[i]];
			assignOp_(dst, data[i], xindices[i]);
		}
	}

private:
	const BASE& xpr_;
	const FuncF assignOp_;
};

}

#endif // EIGEN_SPARSE_TENSOR_ASSIGN_H
