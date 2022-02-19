#include <set>

#include "unsupported/Eigen/CXX11/Tensor"

#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_REDUCE_H
#define EIGEN_SPARSE_TENSOR_REDUCE_H

namespace Eigen
{

template <typename BASE, typename DIM, typename REDUCE_OP>
struct SparseTensorReduceOp
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorReduceOp (const BASE& expr, const std::set<DIM>& dims,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		xpr_(expr), dims_(dims), threshold_(threshold)
	{
		const Dimensions& input_dims = expr.dimensions();
		std::vector<Index> reverse_shuffle(NumDims, -1);
		int j = 0;
		is_identity_ = true;
		in_nelems_ = 1;
		out_nelems_ = 1;
		for (int i = 0; i < NumDims; ++i)
		{
			if (dims.find(i) == dims.end())
			{
				reverse_shuffle[i] = j;
				dimensions_[j] = input_dims[i];
				out_nelems_ *= input_dims[i];
				++j;
				if (is_identity_ && reverse_shuffle[i] != i)
				{
					is_identity_ = false;
				}
			}
			in_nelems_ *= input_dims[i];
		}
		Dimensions rdimensions = dimensions_;
		for (int i = 0; i < NumDims; ++i)
		{
			if (reverse_shuffle[i] == -1)
			{
				reverse_shuffle[i] = j;
				dimensions_[j] = 1;
				rdimensions[j] = input_dims[i];
				++j;
				// whether the shuffle is identity is already determined
			}
		}

		std::array<Index,NumDims> unshuffled_outstrides;
		unshuffled_outstrides[0] = 1;
		instrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			unshuffled_outstrides[i] = unshuffled_outstrides[i - 1] * rdimensions[i - 1];
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
		if (n == 0 || dims_.empty())
		{
			block.set_nnz(n);
			auto result = block.get_data();
			auto rindices = block.get_indices();
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
		[this,&indices](const size_t& l, const size_t& r)
		{
			return (indices[l] % out_nelems_) < (indices[r] % out_nelems_);
		});

		size_t m = 0;
		size_t sidx = sindices[0];
		Scalar tmp_values[n];
		size_t tmp_indices[n];
		Index reduce_size = in_nelems_ / out_nelems_;
		tmp_indices[m] = indices[sidx] % out_nelems_;
		auto accum = reducer_.initialize(data[sidx], indices[sidx] / out_nelems_);
		for (size_t i = 1; i < n; ++i)
		{
			sidx = sindices[i];
			size_t outidx = indices[sidx] % out_nelems_;
			size_t dimidx = indices[sidx] / out_nelems_;
			if (tmp_indices[m] == outidx)
			{
				reducer_.accum(accum, {data[sidx], dimidx});
			}
			else
			{
				const Scalar val = reducer_.finalize(accum, reduce_size);
				if (not_close(val, threshold_))
				{
					tmp_values[m++] = val;
				}
				tmp_indices[m] = outidx;
				accum = reducer_.initialize(data[sidx], dimidx);
			}
		}
		const Scalar val = reducer_.finalize(accum, reduce_size);
		if (not_close(val, threshold_))
		{
			tmp_values[m++] = val;
		}

		block.set_nnz(m);
		auto result = block.get_data();
		auto rindices = block.get_indices();
		std::copy(tmp_values, tmp_values + m, result);
		std::copy(tmp_indices, tmp_indices + m, rindices);
	}

private:
	inline Index dst_index (Index index) const
	{
		if (is_identity_)
		{
			return index;
		}
		Index output_index = 0;
		for (int i = NumDims - 1; i > 0; --i)
		{
			const Index idx = index / fast_instrides_[i];
			output_index += idx * outstrides_[i];
			index -= idx * instrides_[i];
		}
		return (output_index + index * outstrides_[0]);
	}

	const BASE& xpr_;
	const std::set<DIM> dims_;
	REDUCE_OP reducer_;
	Dimensions dimensions_;
	Index in_nelems_;
	Index out_nelems_;
	const Scalar threshold_;

	bool is_identity_;
	std::array<Index, NumDims> instrides_;
	std::array<Index, NumDims> outstrides_;
	std::array<::Eigen::internal::TensorIntDivisor<Index>,NumDims> fast_instrides_;
};

template <typename T, typename STATE>
struct iReducer
{
	virtual ~iReducer (void) = default;

	virtual inline STATE initialize (T value, size_t) const = 0;

	virtual inline void accum (STATE& accum, const std::pair<T,size_t>& validx) const = 0;

	virtual inline T finalize (const STATE& accum, size_t nreduced) const = 0;
};

template <typename T>
struct SumReducer final : public iReducer<T,T>
{
	typedef T State;

	inline T initialize (T value, size_t) const override
	{
		return value;
	}

	inline void accum (State& accum, const std::pair<T,size_t>& validx) const override
	{
		accum += validx.first;
	}

	inline T finalize (const T& accum, size_t) const override
	{
		return accum;
	}
};

template <typename T>
struct ProdReducer final : public iReducer<T,std::pair<T,size_t>>
{
	typedef std::pair<T,size_t> State;

	inline State initialize (T value, size_t) const override
	{
		return {value, 1};
	}

	inline void accum (State& accum, const std::pair<T,size_t>& validx) const override
	{
		accum.first *= validx.first;
		++accum.second;
	}

	inline T finalize (const State& accum, size_t nreduced) const override
	{
		if (accum.second != nreduced)
		{
			return 0; // not every element is visited so we're multiplying by zero
		}
		return accum.first;
	}
};

template <typename T>
struct MaxReducer final : public iReducer<T,std::pair<T,size_t>>
{
	typedef std::pair<T,size_t> State;

	inline State initialize (T value, size_t) const override
	{
		return {value, 1};
	}

	inline void accum (State& accum, const std::pair<T,size_t>& validx) const override
	{
		accum.first = std::max(accum.first, validx.first);
		++accum.second;
	}

	inline T finalize (const State& accum, size_t nreduced) const override
	{
		if (accum.second != nreduced)
		{
			return std::max((T)0, accum.first); // not every element is visited so we consider 0
		}
		return accum.first;
	}
};

template <typename T>
struct MinReducer final : public iReducer<T,std::pair<T,size_t>>
{
	typedef std::pair<T,size_t> State;

	inline State initialize (T value, size_t) const override
	{
		return {value, 1};
	}

	inline void accum (State& accum, const std::pair<T,size_t>& validx) const override
	{
		accum.first = std::min(accum.first, validx.first);
		++accum.second;
	}

	inline T finalize (const State& accum, size_t nreduced) const override
	{
		if (accum.second != nreduced)
		{
			return std::min((T)0, accum.first); // not every element is visited so we consider 0
		}
		return accum.first;
	}
};

template <typename T>
struct ArgMaxReducer final : public iReducer<T,std::pair<T,size_t>>
{
	typedef std::pair<T,size_t> State;

	inline State initialize (T value, size_t idx) const override
	{
		return {value, idx};
	}

	inline void accum (State& accum, const State& validx) const override
	{
		if (accum.first < validx.first)
		{
			accum = validx;
		}
	}

	inline T finalize (const State& accum, size_t nreduced) const override
	{
		return accum.second;
	}
};

}

#endif // EIGEN_SPARSE_TENSOR_REDUCE_H
