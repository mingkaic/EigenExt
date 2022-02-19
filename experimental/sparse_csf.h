#include <numeric>
#include <vector>
#include <utility>

#include "EigenExt/sparse_shape.h"

#ifndef EIGEN_SPARSE_TENSOR_CSF_H
#define EIGEN_SPARSE_TENSOR_CSF_H

namespace Eigen
{

namespace csf
{

struct DimInfo final
{
	std::vector<size_t> nz_counts_;

	std::vector<size_t> nz_indices_;
};

template <size_t RANK>
struct ShapeInfo final
{
	template <typename INDEX>
	ShapeInfo (const ShapeT<RANK>& shape, const INDEX* indices, size_t nnz) :
		nnz_(nnz)
	{
		std::vector<size_t> sindices(indices, indices + nnz);
		size_t nsindices = nnz_;
		for (size_t i = 0; i < RANK; ++i)
		{
			auto& nz_indices = nz_info_[i].nz_indices_;
			auto& nz_counts = nz_info_[i].nz_counts_;
			nz_counts.push_back(0);

			int64_t last_yidx = -1;
			size_t next_nsindices = 0;
			for (size_t j = 0; j < nsindices; ++j)
			{
				auto sidx = sindices[j];
				nz_indices.push_back(sidx % shape[i]);

				int64_t yidx = sidx / shape[i];
				if (yidx != last_yidx)
				{
					sindices[next_nsindices++] = yidx;
					nz_counts.push_back(nz_counts.back() + 1);
				}
				else
				{
					++nz_counts.back();
				}
				last_yidx = yidx;
			}
			nsindices = next_nsindices;
		}
	}

	DimInfo& operator[] (size_t index) { return nz_info_[index]; }

	const DimInfo& operator[] (size_t index) const { return nz_info_[index]; }

	size_t non_zeros (void) const { return nnz_; }

	std::vector<size_t> decode (const ShapeT<RANK>& shape) const
	{
		std::vector<size_t> sindices;
		sindices.reserve(nnz_);
		sindices.push_back(0);
		size_t dimsize = internal::array_prod(shape);
		for (int64_t i = RANK-1; i >= 0; --i)
		{
			dimsize /= shape[i];

			auto& sinfo = nz_info_[i];
			auto& nz_counts = sinfo.nz_counts_;
			auto& nz_indices = sinfo.nz_indices_;

			std::vector<size_t> next_sindices;
			next_sindices.reserve(nz_indices.size());
			for (size_t j = 0, n = nz_counts.size() - 1; j < n; ++j)
			{
				for (size_t k = nz_counts[j], m = nz_counts[j+1]; k < m; ++k)
				{
					next_sindices.push_back(
						nz_indices[k] * dimsize + sindices[j]);
				}
			}
			sindices.swap(next_sindices);
		}
		return sindices;
	}

	std::array<DimInfo,RANK> nz_info_;

	size_t nnz_;
};

}

}

#endif // EIGEN_SPARSE_TENSOR_CSF_H
