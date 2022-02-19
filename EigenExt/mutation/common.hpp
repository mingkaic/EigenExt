
#include <random>

#include "exam/exam.hpp"

#include "muta/mutator.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


template <typename T, size_t RANK>
struct TensorMutator : public muta::Mutator
{
	virtual ~TensorMutator (void) = default;

	using SrcT = SparseTensorSrc<T,RANK>;

	typename Eigen::Tensor<T,RANK>::Dimensions get_shape (const std::string& id)
	{
		auto ints = generate_ints(id, RANK, 1, 7);
		typename Eigen::Tensor<T,RANK>::Dimensions out;
		std::copy(ints.begin(), ints.end(), out.begin());
		return out;
	}

	Tensor<T,RANK> get_sparse_tensor (std::vector<T>& data, std::vector<size_t>& indices,
		const std::string& id, const typename Eigen::Tensor<T,RANK>::Dimensions& shape,
		double density)
	{
		assert(density <= 1);
		size_t n = internal::array_prod(shape);
		assert(n > 0);
		size_t nnz = n * density;

		auto permute = permute_indices(id + "_indices", n, nnz);
		auto decs = generate_decs(id + "_data", nnz, 0, 1);

		std::sort(permute.begin(), permute.end());
		write_entry(id + "_indices", permute);

		indices = std::vector<size_t>(permute.begin(), permute.end());
		data = std::vector<T>(decs.begin(), decs.end());

		Tensor<T,RANK> out(shape);
		out.setZero();
		auto ptr = out.data();
		for (size_t i = 0; i < nnz; ++i)
		{
			ptr[indices[i]] = data[i];
		}
		return out;
	}

	template <size_t CONTRACT_R>
	Tensor<T,RANK> get_rcontract_sparse_tensor (array<IndexPair<int>,CONTRACT_R>& dims,
		std::vector<T>& data, std::vector<size_t>& indices, const std::string& id,
		const Tensor<T,RANK>& ltensor, double density)
	{
		auto ldims = ltensor.dimensions();

		auto lindex = permute_indices("left_dims", RANK, CONTRACT_R);
		auto rindex = permute_indices("right_dims", RANK, CONTRACT_R);

		auto rshape = get_shape("rshape");
		for (size_t i = 0; i < CONTRACT_R; ++i)
		{
			dims[i] = {lindex[i], rindex[i]};
			rshape[rindex[i]] = ldims[lindex[i]];
		}
		write_entry("rshape", std::vector<int64_t>(rshape.begin(), rshape.end()));
		return get_sparse_tensor(data, indices, id, rshape, density);
	}

	SrcT make_sparse_src (const Tensor<T,RANK>& tens,
		const std::vector<T>& data, const std::vector<size_t>& indices)
	{
		auto dims = tens.dimensions();
		typedef typename SparseTensorSrc<T,RANK>::Dimensions Shape;
		Shape shape;
		std::copy(dims.begin(), dims.end(), shape.begin());
		return SrcT(shape, indices.size(), indices.data(), data.data());
	}
};
