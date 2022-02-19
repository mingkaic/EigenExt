#include <list>
#include <unordered_map>

#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_CONTRACT_H
#define EIGEN_SPARSE_TENSOR_CONTRACT_H

namespace Eigen
{

template <typename CIDX>
using ContractDimsT = std::vector<std::pair<CIDX,CIDX>>;

template <typename LBASE, typename RBASE, typename CIDX>
struct SparseTensorContractOp final
{
	typedef typename LBASE::Scalar Scalar;
	typedef typename LBASE::Index Index;
	typedef typename LBASE::Dimensions Dimensions;
	static const size_t NumDims = LBASE::NumDims;

	inline SparseTensorContractOp (
		const LBASE& lxpr, const RBASE& rxpr, const ContractDimsT<CIDX>& dims,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		dims_(dims), lxpr_(lxpr), rxpr_(rxpr), threshold_(threshold)
	{
		auto ldims = lxpr_.dimensions();
		auto rdims = rxpr_.dimensions();
		linstrides_[0] = rinstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			linstrides_[i] = linstrides_[i - 1] * ldims[i - 1];
			rinstrides_[i] = rinstrides_[i - 1] * rdims[i - 1];
		}
		size_t ndims = dims_.size();
		std::fill(lcstrides_.begin(), lcstrides_.end(), 0);
		std::fill(rcstrides_.begin(), rcstrides_.end(), 0);
		lcstrides_[dims_[0].first] = rcstrides_[dims_[0].second] = 1;
		for (int i = 1; i < ndims; ++i)
		{
			auto& prevdims = dims_[i - 1];
			lcstrides_[dims_[i].first] = lcstrides_[prevdims.first] * ldims[prevdims.first];
			rcstrides_[dims_[i].second] = rcstrides_[prevdims.second] * rdims[prevdims.second];
		}
		routstrides_[0] = lmidstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			if (rcstrides_[i - 1] > 0)
			{
				routstrides_[i] = routstrides_[i - 1];
			}
			else
			{
				routstrides_[i] = routstrides_[i - 1] * rdims[i - 1];
			}
			if (lcstrides_[i - 1] > 0)
			{
				lmidstrides_[i] = lmidstrides_[i - 1];
			}
			else
			{
				lmidstrides_[i] = lmidstrides_[i - 1] * ldims[i - 1];
			}
		}
		if (rcstrides_[NumDims - 1] > 0)
		{
			loutstrides_[0] = std::max(routstrides_[NumDims - 1], (Index) 1);
		}
		else
		{
			loutstrides_[0] = routstrides_[NumDims - 1] * rdims[NumDims - 1];
		}
		for (int i = 1; i < NumDims; ++i)
		{
			loutstrides_[i] = loutstrides_[i - 1] * loutstrides_[0];
		}
	}

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto ldata = lxpr_.data();
		auto lxindices = lxpr_.get_indices();
		auto rdata = rxpr_.data();
		auto rxindices = rxpr_.get_indices();

		std::unordered_map<Index,std::vector<std::pair<Index,size_t>>> clefts;
		for (size_t i = 0, n = lxindices.size(); i < n; ++i)
		{
			Index interm_index;
			auto lindex = ldst_index(interm_index, lxindices[i]);
			clefts[interm_index].push_back({lindex, i});
		}
		auto rn = loutstrides_[0];
		std::unordered_map<Index,Scalar> buffer;
		std::list<Index> bindices;
		for (size_t i = 0, n = rxindices.size(); i < n; ++i)
		{
			Index interm_index;
			auto rindex = rdst_index(interm_index, rxindices[i]);
			auto clit = clefts.find(interm_index);
			if (clit != clefts.end())
			{
				for (auto& left : clit->second)
				{
					const Index key = rindex + left.first * rn;
					const Scalar val = ldata[left.second] * rdata[i];
					auto it = buffer.find(key);
					if (it == buffer.end())
					{
						bindices.push_back(key);
						buffer.emplace(key, val);
					}
					else
					{
						it->second += val;
					}
				}
			}
		}
		bindices.sort();
		block.allocate(buffer.size());
		auto result = block.get_data();
		auto rindices = block.get_indices();
		size_t nnz = 0;
		for (auto index : bindices)
		{
			if (not_close(buffer[index], threshold_))
			{
				rindices[nnz] = index;
				result[nnz] = buffer[index];
				++nnz;
			}
		}
		block.set_nnz(nnz);
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		block.setZero();
		auto ldata = lxpr_.data();
		auto lxindices = lxpr_.get_indices();
		auto rdata = rxpr_.data();
		auto rxindices = rxpr_.get_indices();

		std::unordered_map<Index,std::vector<std::pair<Index,size_t>>> clefts;
		for (size_t i = 0, n = lxindices.size(); i < n; ++i)
		{
			Index interm_index;
			auto lindex = ldst_index(interm_index, lxindices[i]);
			clefts[interm_index].push_back({lindex, i});
		}
		auto rn = loutstrides_[0];
		auto result = block.data();
		for (size_t i = 0, n = rxindices.size(); i < n; ++i)
		{
			Index interm_index;
			auto rindex = rdst_index(interm_index, rxindices[i]);
			auto clit = clefts.find(interm_index);
			if (clit != clefts.end())
			{
				for (auto& left : clit->second)
				{
					result[rindex + left.first * rn] += ldata[left.second] * rdata[i];
				}
			}
		}
	}

private:
	// Return left index, and updates contraction index
	inline Index ldst_index (Index& interm_index, Index index) const
	{
		interm_index = 0;
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index iidx = index / linstrides_[i];
			auto lcs = lcstrides_[i];
			if (lcs > 0)
			{
				interm_index += iidx * lcs;
			}
			else
			{
				output_index += iidx * lmidstrides_[i];
			}
			index -= iidx * linstrides_[i];
		}
		return output_index;
	}

	// Return output index assuming left index of 0, and updates contraction index
	inline Index rdst_index (Index& interm_index, Index index) const
	{
		interm_index = 0;
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index iidx = index / rinstrides_[i];
			auto rcs = rcstrides_[i];
			if (rcs > 0)
			{
				interm_index += iidx * rcs;
			}
			else
			{
				output_index += iidx * routstrides_[i];
			}
			index -= iidx * rinstrides_[i];
		}
		return output_index;
	}

	Scalar threshold_;

	const ContractDimsT<CIDX> dims_;
	const LBASE& lxpr_;
	const RBASE& rxpr_;

	std::array<Index,NumDims> linstrides_;
	std::array<Index,NumDims> rinstrides_;
	std::array<Index,NumDims> loutstrides_;
	std::array<Index,NumDims> lmidstrides_;
	std::array<Index,NumDims> routstrides_;
	std::array<Index,NumDims> lcstrides_;
	std::array<Index,NumDims> rcstrides_;
};

template <typename BASE, typename DENSE, typename CIDX>
struct SparseTensorContractOp<BASE,::Eigen::TensorMap<DENSE>,CIDX> final
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorContractOp (
		const BASE& lxpr, const ::Eigen::TensorMap<DENSE>& rxpr, const ContractDimsT<CIDX>& dims,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		dims_(dims), lxpr_(lxpr), rxpr_(rxpr)
	{
		auto ldims = lxpr_.dimensions();
		auto rdims = rxpr_.dimensions();
		linstrides_[0] = rinstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			linstrides_[i] = linstrides_[i - 1] * ldims[i - 1];
			rinstrides_[i] = rinstrides_[i - 1] * rdims[i - 1];
		}
		size_t ndims = dims_.size();
		std::fill(lcstrides_.begin(), lcstrides_.end(), 0);
		std::fill(rcstrides_.begin(), rcstrides_.end(), 0);
		lcstrides_[dims_[0].first] = rcstrides_[dims_[0].second] = 1;
		for (int i = 1; i < ndims; ++i)
		{
			auto& prevdims = dims_[i - 1];
			lcstrides_[dims_[i].first] = lcstrides_[prevdims.first] * ldims[prevdims.first];
			rcstrides_[dims_[i].second] = rcstrides_[prevdims.second] * rdims[prevdims.second];
		}
		routstrides_[0] = lmidstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			if (rcstrides_[i - 1] > 0)
			{
				routstrides_[i] = routstrides_[i - 1];
			}
			else
			{
				routstrides_[i] = routstrides_[i - 1] * rdims[i - 1];
			}
			if (lcstrides_[i - 1] > 0)
			{
				lmidstrides_[i] = lmidstrides_[i - 1];
			}
			else
			{
				lmidstrides_[i] = lmidstrides_[i - 1] * ldims[i - 1];
			}
		}
		if (rcstrides_[NumDims - 1] > 0)
		{
			loutstrides_[0] = std::max(routstrides_[NumDims - 1], (Index) 1);
		}
		else
		{
			loutstrides_[0] = routstrides_[NumDims - 1] * rdims[NumDims - 1];
		}
		for (int i = 1; i < NumDims; ++i)
		{
			loutstrides_[i] = loutstrides_[i - 1] * loutstrides_[0];
		}

		cstrides_.push_back(1);
		for (int i = 1; i < ndims; ++i)
		{
			cstrides_.push_back(cstrides_.back() * rdims[dims[i-1].second]);
		}
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		block.setZero();
		auto ldata = lxpr_.data();
		auto lxindices = lxpr_.get_indices();
		auto rdata = rxpr_.data();

		auto result = block.data();
		auto rn = loutstrides_[0];
		for (size_t i = 0, n = lxindices.size(); i < n; ++i)
		{
			Index interm_index;
			auto lindex = ldst_index(interm_index, lxindices[i]);
			for (Index j = 0; j < rn; ++j)
			{
				result[j + lindex * rn] += ldata[i] * rdata[rsrc_index(j, interm_index)];
			}
		}
	}

private:
	// Return left index, and updates contraction index
	inline Index ldst_index (Index& interm_index, Index index) const
	{
		interm_index = 0;
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index iidx = index / linstrides_[i];
			auto lcs = lcstrides_[i];
			if (lcs > 0)
			{
				interm_index += iidx * lcs;
			}
			else
			{
				output_index += iidx * lmidstrides_[i];
			}
			index -= iidx * linstrides_[i];
		}
		return output_index;
	}

	inline Index rsrc_index (Index index, Index interm_index) const
	{
		Index input_index = 0;
		for (int i = cstrides_.size() - 1; i >= 0; --i)
		{
			const Index oidx = interm_index / cstrides_[i];
			input_index += oidx * rinstrides_[dims_[i].second];
			interm_index -= oidx * cstrides_[i];
		}
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index oidx = index / routstrides_[i];
			if (rcstrides_[i] < 1)
			{
				input_index += oidx * rinstrides_[i];
			}
			index -= oidx * routstrides_[i];
		}
		return input_index;
	}

	const ContractDimsT<CIDX> dims_;
	const BASE& lxpr_;
	const ::Eigen::TensorMap<DENSE>& rxpr_;

	std::array<Index,NumDims> linstrides_;
	std::array<Index,NumDims> rinstrides_;
	std::array<Index,NumDims> loutstrides_;
	std::array<Index,NumDims> lmidstrides_;
	std::array<Index,NumDims> routstrides_;
	std::array<Index,NumDims> lcstrides_;
	std::array<Index,NumDims> rcstrides_;
	std::vector<Index> cstrides_;
};

template <typename DENSE, typename BASE, typename CIDX>
struct SparseTensorContractOp<::Eigen::TensorMap<DENSE>,BASE,CIDX> final
{
	typedef typename BASE::Scalar Scalar;
	typedef typename BASE::Index Index;
	typedef typename BASE::Dimensions Dimensions;
	static const size_t NumDims = BASE::NumDims;

	inline SparseTensorContractOp (
		const ::Eigen::TensorMap<DENSE>& lxpr, const BASE& rxpr, const ContractDimsT<CIDX>& dims,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		dims_(dims), lxpr_(lxpr), rxpr_(rxpr)
	{
		auto ldims = lxpr_.dimensions();
		auto rdims = rxpr_.dimensions();
		rinstrides_[0] = linstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			linstrides_[i] = linstrides_[i - 1] * ldims[i - 1];
			rinstrides_[i] = rinstrides_[i - 1] * rdims[i - 1];
		}
		size_t ndims = dims_.size();
		std::fill(rcstrides_.begin(), rcstrides_.end(), 0);
		std::fill(lcstrides_.begin(), lcstrides_.end(), 0);
		rcstrides_[dims_[0].second] = lcstrides_[dims_[0].first] = 1;
		for (int i = 1; i < ndims; ++i)
		{
			auto& prevdims = dims_[i - 1];
			rcstrides_[dims_[i].second] = rcstrides_[prevdims.second] * rdims[prevdims.second];
			lcstrides_[dims_[i].first] = lcstrides_[prevdims.first] * ldims[prevdims.first];
		}
		loutstrides_[0] = rmidstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			if (lcstrides_[i - 1] > 0)
			{
				loutstrides_[i] = loutstrides_[i - 1];
			}
			else
			{
				loutstrides_[i] = loutstrides_[i - 1] * ldims[i - 1];
			}
			if (rcstrides_[i - 1] > 0)
			{
				rmidstrides_[i] = rmidstrides_[i - 1];
			}
			else
			{
				rmidstrides_[i] = rmidstrides_[i - 1] * rdims[i - 1];
			}
		}
		if (lcstrides_[NumDims - 1] > 0)
		{
			routstrides_[0] = std::max(loutstrides_[NumDims - 1], (Index) 1);
		}
		else
		{
			routstrides_[0] = loutstrides_[NumDims - 1] * ldims[NumDims - 1];
		}
		for (int i = 1; i < NumDims; ++i)
		{
			routstrides_[i] = routstrides_[i - 1] * routstrides_[0];
		}

		cstrides_.push_back(1);
		for (int i = 1; i < ndims; ++i)
		{
			cstrides_.push_back(cstrides_.back() * ldims[dims[i-1].second]);
		}
		rn_ = 1;
		for (int i = 0; i < NumDims; ++i)
		{
			if (rcstrides_[i] < 1)
			{
				rn_ *= rdims[i];
			}
		}
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		block.setZero();
		auto ldata = rxpr_.data();
		auto lxindices = rxpr_.get_indices();
		auto rdata = lxpr_.data();

		auto result = block.data();
		auto ln = routstrides_[0];
		for (size_t i = 0, n = lxindices.size(); i < n; ++i)
		{
			Index interm_index;
			auto lindex = rdst_index(interm_index, lxindices[i]);
			for (Index j = 0; j < ln; ++j)
			{
				result[lindex + j * rn_] += ldata[i] * rdata[lsrc_index(j, interm_index)];
			}
		}
	}

private:
	inline Index rdst_index (Index& interm_index, Index index) const
	{
		interm_index = 0;
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index iidx = index / rinstrides_[i];
			auto lcs = rcstrides_[i];
			if (lcs > 0)
			{
				interm_index += iidx * lcs;
			}
			else
			{
				output_index += iidx * rmidstrides_[i];
			}
			index -= iidx * rinstrides_[i];
		}
		return output_index;
	}

	inline Index lsrc_index (Index index, Index interm_index) const
	{
		Index input_index = 0;
		for (int i = cstrides_.size() - 1; i >= 0; --i)
		{
			const Index oidx = interm_index / cstrides_[i];
			input_index += oidx * linstrides_[dims_[i].first];
			interm_index -= oidx * cstrides_[i];
		}
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index oidx = index / loutstrides_[i];
			if (lcstrides_[i] < 1)
			{
				input_index += oidx * linstrides_[i];
			}
			index -= oidx * loutstrides_[i];
		}
		return input_index;
	}

	const ContractDimsT<CIDX> dims_;
	const ::Eigen::TensorMap<DENSE>& lxpr_;
	const BASE& rxpr_;

	std::array<Index,NumDims> linstrides_;
	std::array<Index,NumDims> rinstrides_;
	std::array<Index,NumDims> loutstrides_;
	std::array<Index,NumDims> rmidstrides_;
	std::array<Index,NumDims> routstrides_;
	std::array<Index,NumDims> lcstrides_;
	std::array<Index,NumDims> rcstrides_;
	std::vector<Index> cstrides_;
	Index rn_;
};

}

#endif // EIGEN_SPARSE_TENSOR_CONTRACT_H
