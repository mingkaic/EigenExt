#include "EigenExt/sparse_util.h"
#include "EigenExt/sparse_tensor_dst.h"

#ifndef EIGEN_SPARSE_TENSOR_CONVOLVE_H
#define EIGEN_SPARSE_TENSOR_CONVOLVE_H

namespace Eigen
{

template <typename IBASE, typename KBASE, typename DIM>
struct SparseTensorConvolveOp
{
	typedef typename IBASE::Scalar Scalar;
	typedef typename IBASE::Index Index;
	typedef typename IBASE::Dimensions Dimensions;
	static const size_t NumDims = IBASE::NumDims;

	inline SparseTensorConvolveOp (const IBASE& iExpr, const KBASE& kExpr,
		const std::array<DIM,NumDims>& dims,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		ixpr_(iExpr), kxpr_(kExpr), dims_(dims),
		idimensions_(ixpr_.dimensions()), threshold_(threshold)
	{
		auto kdims = kxpr_.dimensions();

		Dimensions dimensions;
		nout_ = 1;
		kern_identity_ = true;
		for (int i = 0; i < NumDims; ++i)
		{
			DIM outdim = dims[i];
			dimensions[outdim] = idimensions_[outdim] - kdims[i] + 1;
			nout_ *= dimensions[outdim];
			kdimensions_[dims[i]] = kdims[i];

			if (kern_identity_ && outdim != i)
			{
				kern_identity_ = false;
			}
		}
		std::array<Index,NumDims> shuffled_koutstride;
		shuffled_koutstride[0] = 1;
		instrides_[0] = 1;
		outstrides_[0] = 1;
		kinstrides_[0] = 1;
		koutstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			shuffled_koutstride[i] = shuffled_koutstride[i - 1] * kdims[dims[i]];
			instrides_[i] = instrides_[i - 1] * idimensions_[i - 1];
			outstrides_[i] = outstrides_[i - 1] * dimensions[i - 1];
			kinstrides_[i] = kinstrides_[i - 1] * kdims[i - 1];
			koutstrides_[i] = koutstrides_[i - 1] * kdimensions_[i - 1];
		}

		for (int i = 0; i < NumDims; ++i)
		{
			kmidstrides_[i] = shuffled_koutstride[dims[i]];
		}
	}

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto iindices = ixpr_.get_indices();
		auto kindices = kxpr_.get_indices();
		auto idata = ixpr_.data();
		auto kdata = kxpr_.data();

		std::unordered_map<Index,Scalar> buffer;
		std::list<Index> bindices;
		for (size_t k = 0, m = kindices.size(); k < m; ++k)
		{
			size_t kdst = kdst_index(kindices[k]);
			for (size_t i = 0, n = iindices.size(); i < n; ++i)
			{
				const Index dstidx = dst_index(iindices[i], kdst, nout_);
				if (dstidx < nout_)
				{
					const Scalar val = idata[i] * kdata[k];
					auto it = buffer.find(dstidx);
					if (it == buffer.end())
					{
						bindices.push_back(dstidx);
						buffer.emplace(dstidx, val);
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
		auto iindices = ixpr_.get_indices();
		auto kindices = kxpr_.get_indices();
		auto idata = ixpr_.data();
		auto kdata = kxpr_.data();

		auto result = block.data();
		for (size_t k = 0, m = kindices.size(); k < m; ++k)
		{
			size_t kdst = kdst_index(kindices[k]);
			for (size_t i = 0, n = iindices.size(); i < n; ++i)
			{
				const Index dstidx = dst_index(iindices[i], kdst, nout_);
				if (dstidx < nout_)
				{
					result[dstidx] += idata[i] * kdata[k];
				}
			}
		}
	}

private:
	inline Index kdst_index (Index kindex) const
	{
		Index kmid_index = 0;
		if (kern_identity_)
		{
			kmid_index = kindex;
		}
		else
		{
			for (int i = NumDims - 1; i >= 0; --i)
			{
				const Index kidx = kindex / kinstrides_[i];
				kmid_index += kidx * kmidstrides_[i];
				kindex -= kidx * kinstrides_[i];
			}
		}
		return kmid_index;
	}

	inline Index dst_index (Index iindex, Index kindex, Index fallback) const
	{
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index iidx = iindex / instrides_[i];
			const Index kidx = kindex / koutstrides_[i];

			if (iidx < kidx || (iidx + kdimensions_[i] - kidx) > idimensions_[i])
			{
				return fallback;
			}
			output_index += (iidx - kidx) * outstrides_[i];

			iindex -= iidx * instrides_[i];
			kindex -= kidx * koutstrides_[i];
		}
		return output_index;
	}

	const IBASE& ixpr_;
	const KBASE& kxpr_;
	const std::array<DIM,NumDims> dims_;
	const Dimensions idimensions_;
	const Scalar threshold_;
	Index nout_;
	Dimensions kdimensions_;

	bool kern_identity_;
	std::array<DIM,NumDims> instrides_;
	std::array<DIM,NumDims> outstrides_;
	std::array<DIM,NumDims> kinstrides_;
	std::array<DIM,NumDims> kmidstrides_;
	std::array<DIM,NumDims> koutstrides_;
};

template <typename IBASE, typename KOTHER, typename DIM>
struct SparseTensorConvolveOp<IBASE, ::Eigen::TensorMap<KOTHER>, DIM>
{
	typedef typename IBASE::Scalar Scalar;
	typedef typename IBASE::Index Index;
	typedef typename IBASE::Dimensions Dimensions;
	static const size_t NumDims = IBASE::NumDims;

	inline SparseTensorConvolveOp (const IBASE& iExpr, const ::Eigen::TensorMap<KOTHER>& kExpr,
		const std::array<DIM,NumDims>& dims,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		ixpr_(iExpr), kxpr_(kExpr), idimensions_(ixpr_.dimensions()), dims_(dims), threshold_(threshold)
	{
		auto kdims = kxpr_.dimensions();

		Dimensions dimensions;
		nkin_ = 1;
		nout_ = 1;
		kern_identity_ = true;
		for (int i = 0; i < NumDims; ++i)
		{
			nkin_ *= kdims[i];
			nout_ *= dimensions[i];

			DIM outdim = dims[i];
			dimensions[outdim] = idimensions_[outdim] - kdims[i] + 1;
			kdimensions_[dims[i]] = kdims[i];

			if (kern_identity_ && outdim != i)
			{
				kern_identity_ = false;
			}
		}
		std::array<Index,NumDims> shuffled_koutstride;
		shuffled_koutstride[0] = 1;
		instrides_[0] = 1;
		outstrides_[0] = 1;
		kinstrides_[0] = 1;
		koutstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			shuffled_koutstride[i] = shuffled_koutstride[i - 1] * kdims[dims[i]];
			instrides_[i] = instrides_[i - 1] * idimensions_[i - 1];
			outstrides_[i] = outstrides_[i - 1] * dimensions[i - 1];
			kinstrides_[i] = kinstrides_[i - 1] * kdims[i - 1];
			koutstrides_[i] = koutstrides_[i - 1] * kdimensions_[i - 1];
		}

		for (int i = 0; i < NumDims; ++i)
		{
			kmidstrides_[i] = shuffled_koutstride[dims[i]];
		}
	}

	template <typename INDEX>
	void write (iSparseTensorDst<Scalar,INDEX>& block) const
	{
		auto iindices = ixpr_.get_indices();
		auto idata = ixpr_.data();
		auto kdata = kxpr_.data();

		std::unordered_map<Index,Scalar> buffer;
		std::list<Index> bindices;
		for (size_t k = 0; k < nkin_; ++k)
		{
			size_t kdst = kdst_index(k);
			for (size_t i = 0, n = iindices.size(); i < n; ++i)
			{
				const Index dstidx = dst_index(iindices[i], kdst, nout_);
				if (dstidx < nout_)
				{
					const Scalar val = idata[i] * kdata[k];
					auto it = buffer.find(dstidx);
					if (it == buffer.end())
					{
						bindices.push_back(dstidx);
						buffer.emplace(dstidx, val);
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
		auto iindices = ixpr_.get_indices();
		auto kdata = kxpr_.data();
		auto idata = ixpr_.data();

		auto result = block.data();
		for (size_t k = 0; k < nkin_; ++k)
		{
			size_t kdst = kdst_index(k);
			for (size_t i = 0, n = iindices.size(); i < n; ++i)
			{
				const Index dstidx = dst_index(iindices[i], kdst, nout_);
				if (dstidx < nout_)
				{
					result[dstidx] += idata[i] * kdata[k];
				}
			}
		}
	}

private:
	inline Index kdst_index (Index kindex) const
	{
		Index kmid_index = 0;
		if (kern_identity_)
		{
			kmid_index = kindex;
		}
		else
		{
			for (int i = NumDims - 1; i >= 0; --i)
			{
				const Index kidx = kindex / kinstrides_[i];
				kmid_index += kidx * kmidstrides_[i];
				kindex -= kidx * kinstrides_[i];
			}
		}
		return kmid_index;
	}

	inline Index dst_index (Index iindex, Index kindex, Index fallback) const
	{
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index iidx = iindex / instrides_[i];
			const Index kidx = kindex / koutstrides_[i];

			if (iidx < kidx || (iidx + kdimensions_[i] - kidx) > idimensions_[i])
			{
				return fallback;
			}
			output_index += (iidx - kidx) * outstrides_[i];

			iindex -= iidx * instrides_[i];
			kindex -= kidx * koutstrides_[i];
		}
		return output_index;
	}

	const IBASE& ixpr_;
	const ::Eigen::TensorMap<KOTHER>& kxpr_;
	const std::array<DIM,NumDims> dims_;
	const Dimensions idimensions_;
	const Scalar threshold_;
	Index nkin_;
	Index nout_;
	Dimensions kdimensions_;

	bool kern_identity_;
	std::array<DIM,NumDims> instrides_;
	std::array<DIM,NumDims> outstrides_;
	std::array<DIM,NumDims> kinstrides_;
	std::array<DIM,NumDims> kmidstrides_;
	std::array<DIM,NumDims> koutstrides_;
};

template <typename IOTHER, typename KBASE, typename DIM>
struct SparseTensorConvolveOp<::Eigen::TensorMap<IOTHER>, KBASE, DIM>
{
	typedef typename KBASE::Scalar Scalar;
	typedef typename KBASE::Index Index;
	typedef typename KBASE::Dimensions Dimensions;
	static const size_t NumDims = KBASE::NumDims;

	inline SparseTensorConvolveOp (const ::Eigen::TensorMap<IOTHER>& iExpr, const KBASE& kExpr,
		const std::array<DIM,NumDims>& dims,
		const Scalar& threshold = std::numeric_limits<Scalar>::epsilon()) :
		ixpr_(iExpr), kxpr_(kExpr), dims_(dims),
		idimensions_(iExpr.dimensions()), threshold_(threshold)
	{
		auto kdims = kxpr_.dimensions();

		Dimensions dimensions;
		nin_ = 1;
		nout_ = 1;
		kern_identity_ = true;
		for (int i = 0; i < NumDims; ++i)
		{
			nin_ *= idimensions_[i];
			nout_ *= dimensions[i];

			DIM outdim = dims[i];
			dimensions[outdim] = idimensions_[outdim] - kdims[i] + 1;
			kdimensions_[dims[i]] = kdims[i];

			if (kern_identity_ && outdim != i)
			{
				kern_identity_ = false;
			}
		}
		std::array<Index,NumDims> shuffled_koutstride;
		shuffled_koutstride[0] = 1;
		instrides_[0] = 1;
		outstrides_[0] = 1;
		kinstrides_[0] = 1;
		koutstrides_[0] = 1;
		for (int i = 1; i < NumDims; ++i)
		{
			shuffled_koutstride[i] = shuffled_koutstride[i - 1] * kdims[dims[i]];
			instrides_[i] = instrides_[i - 1] * idimensions_[i - 1];
			outstrides_[i] = outstrides_[i - 1] * dimensions[i - 1];
			kinstrides_[i] = kinstrides_[i - 1] * kdims[i - 1];
			koutstrides_[i] = koutstrides_[i - 1] * kdimensions_[i - 1];
		}

		for (int i = 0; i < NumDims; ++i)
		{
			kmidstrides_[i] = shuffled_koutstride[dims[i]];
		}
	}

	template <typename OTHER>
	void write (::Eigen::TensorMap<OTHER>& block) const
	{
		block.setZero();
		auto kindices = kxpr_.get_indices();
		auto kdata = kxpr_.data();
		auto idata = ixpr_.data();

		auto result = block.data();
		for (size_t k = 0, m = kindices.size(); k < m; ++k)
		{
			size_t kdst = kdst_index(kindices[k]);
			for (size_t i = 0; i < nin_; ++i)
			{
				const Index dstidx = dst_index(i, kdst, nout_);
				if (dstidx < nout_)
				{
					result[dstidx] += idata[i] * kdata[k];
				}
			}
		}
	}

private:
	inline Index kdst_index (Index kindex) const
	{
		Index kmid_index = 0;
		if (kern_identity_)
		{
			kmid_index = kindex;
		}
		else
		{
			for (int i = NumDims - 1; i >= 0; --i)
			{
				const Index kidx = kindex / kinstrides_[i];
				kmid_index += kidx * kmidstrides_[i];
				kindex -= kidx * kinstrides_[i];
			}
		}
		return kmid_index;
	}

	inline Index dst_index (Index iindex, Index kindex, Index fallback) const
	{
		Index output_index = 0;
		for (int i = NumDims - 1; i >= 0; --i)
		{
			const Index iidx = iindex / instrides_[i];
			const Index kidx = kindex / koutstrides_[i];

			if (iidx < kidx || (iidx + kdimensions_[i] - kidx) > idimensions_[i])
			{
				return fallback;
			}
			output_index += (iidx - kidx) * outstrides_[i];

			iindex -= iidx * instrides_[i];
			kindex -= kidx * koutstrides_[i];
		}
		return output_index;
	}

	const ::Eigen::TensorMap<IOTHER>& ixpr_;
	const KBASE& kxpr_;
	const std::array<DIM,NumDims> dims_;
	const typename ::Eigen::TensorMap<IOTHER>::Dimensions idimensions_;
	const Scalar threshold_;
	Index nin_;
	Index nout_;
	Dimensions kdimensions_;

	bool kern_identity_;
	std::array<DIM,NumDims> instrides_;
	std::array<DIM,NumDims> outstrides_;
	std::array<DIM,NumDims> kinstrides_;
	std::array<DIM,NumDims> kmidstrides_;
	std::array<DIM,NumDims> koutstrides_;
};

}

#endif // EIGEN_SPARSE_TENSOR_CONVOLVE_H
