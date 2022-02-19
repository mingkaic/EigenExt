#include "Eigen/Core"

#ifndef EIGEN_SPARSE_TENSOR_DST_H
#define EIGEN_SPARSE_TENSOR_DST_H

namespace Eigen
{

template <typename T, typename INDEX=size_t>
struct iSparseTensorDst
{
	virtual ~iSparseTensorDst (void) = default;

	virtual T* get_data (void) = 0;

	virtual INDEX* get_indices (void) = 0;

	virtual INDEX non_zeros (void) const = 0;

	virtual size_t alloc_size (void) const = 0;

	virtual void allocate (size_t n) = 0;

	virtual void set_nnz (INDEX nnz) = 0;
};

template <typename T, typename STORAGE>
struct StorageInfo final
{
	using AllocF = std::function<void(STORAGE&,size_t)>;
	using GetDataF = std::function<T*(STORAGE&)>;
	using GetSizeF = std::function<size_t(STORAGE&)>;

	AllocF allocate_;
	GetDataF get_;
	GetSizeF size_;
};

template <typename T>
static void vector_allocate (std::vector<T>& vec, size_t n)
{
	size_t existing = vec.size();
	if (vec.size() < n)
	{
		vec.insert(vec.end(), n - existing, 0);
	}
}

template <typename T>
StorageInfo<T,std::vector<T>> vector_info (void)
{
	return StorageInfo<T,std::vector<T>>{
		vector_allocate<T>,
		[](std::vector<T>& vec)
		{
			return &vec[0];
		},
		[](std::vector<T>& vec)
		{
			return vec.size();
		}
	};
}

template <typename T, typename STORAGE, typename INDEX=size_t>
struct SparseTensorDst final : public iSparseTensorDst<T,INDEX>
{
	typedef T Scalar;
	typedef DimT Index;

	inline SparseTensorDst (INDEX& nnz, STORAGE& data, std::vector<INDEX>& indices,
		const StorageInfo<T,STORAGE>& data_info) :
		nnz_(&nnz), data_(&data), indices_(&indices), dinfo_(data_info) {}

	T* get_data (void) override
	{
		return dinfo_.get_(*data_);
	}

	INDEX* get_indices (void) override
	{
		if (indices_->empty())
		{
			return nullptr;
		}
		return &indices_->at(0);
	}

	INDEX non_zeros (void) const override { return *nnz_; }

	size_t alloc_size (void) const override
	{
		return std::min(dinfo_.size_(*data_), indices_->size());
	}

	void allocate (size_t n) override
	{
		dinfo_.allocate_(*data_, n);
		vector_allocate(*indices_, n);
	}

	void set_nnz (INDEX nnz) override
	{
		allocate(nnz);
		*nnz_ = nnz;
	}

private:
	INDEX* nnz_;

	STORAGE* data_;

	std::vector<INDEX>* indices_;

	StorageInfo<T,STORAGE> dinfo_;
};

}

#endif // EIGEN_SPARSE_TENSOR_DST_H
