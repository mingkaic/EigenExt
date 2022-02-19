
#include "EigenExt/mutation/common.hpp"


template <typename T, size_t RANK>
struct ContractMutator : public TensorMutator<T,RANK>
{
	virtual ~ContractMutator (void) = default;

	template <size_t CONTRACT_R>
	void generate_contract_to_sparse_testcase (double ldensity, double rdensity,
		T eps = std::numeric_limits<T>::epsilon())
	{
		array<IndexPair<int>,CONTRACT_R> dims;
		std::vector<T> adata;
		std::vector<T> bdata;
		std::vector<size_t> aindices;
		std::vector<size_t> bindices;

		auto atens = this->get_sparse_tensor(
			adata, aindices, "left", this->get_shape("lshape"), ldensity);
		auto btens = this->template get_rcontract_sparse_tensor<CONTRACT_R>(
			dims, bdata, bindices, "right", atens, rdensity);

		array<IndexPair<int>,CONTRACT_R> cdims;
		for (size_t i = 0; i < CONTRACT_R; ++i)
		{
			cdims[i] = {dims[i].second, dims[i].first};
		}
		Tensor<T,(RANK-CONTRACT_R)*2> expect = btens.contract(atens, cdims);

		// register expected
		auto expect_ptr = expect.data();
		std::vector<T> expected_data(expect_ptr,
			expect_ptr + internal::array_prod(expect.dimensions()));
		this->write_entry("expected_data", expected_data);

		ContractDimsT<int> contract_dims(dims.size());
		for (size_t i = 0; i < CONTRACT_R; ++i)
		{
			contract_dims[i] = {dims[i].first, dims[i].second};
		}

		size_t nnz = 0;
		std::vector<T> outdata;
		std::vector<size_t> outindex;
		SparseTensorDst<T,std::vector<T>> out(nnz, outdata, outindex, vector_info<T>());

		auto a = this->make_sparse_src(atens, adata, aindices);
		auto b = this->make_sparse_src(btens, bdata, bindices);

		a.contract(b, contract_dims).write(out);

		std::vector<T> got_data(expected_data.size(), 0);
		for (size_t i = 0; i < nnz; ++i)
		{
			got_data[outindex[i]] = outdata[i];
		}
		EXPECT_ARRCLOSE(expected_data, got_data, eps);
	}

	template <size_t CONTRACT_R>
	void generate_contract_to_dense_testcase (double ldensity, double rdensity,
		T eps = std::numeric_limits<T>::epsilon())
	{
		array<IndexPair<int>,CONTRACT_R> dims;
		std::vector<T> adata;
		std::vector<T> bdata;
		std::vector<size_t> aindices;
		std::vector<size_t> bindices;

		auto atens = this->get_sparse_tensor(
			adata, aindices, "left", this->get_shape("lshape"), ldensity);
		auto btens = this->template get_rcontract_sparse_tensor<CONTRACT_R>(
			dims, bdata, bindices, "right", atens, rdensity);

		array<IndexPair<int>,CONTRACT_R> cdims;
		for (size_t i = 0; i < CONTRACT_R; ++i)
		{
			cdims[i] = {dims[i].second, dims[i].first};
		}
		Tensor<T,(RANK-CONTRACT_R)*2> expect = btens.contract(atens, cdims);

		// register expected
		auto expect_ptr = expect.data();
		std::vector<T> expected_data(expect_ptr,
			expect_ptr + internal::array_prod(expect.dimensions()));
		this->write_entry("expected_data", expected_data);

		ContractDimsT<int> contract_dims(dims.size());
		for (size_t i = 0; i < CONTRACT_R; ++i)
		{
			contract_dims[i] = {dims[i].first, dims[i].second};
		}

		std::vector<T> got_data(expected_data.size(), 0);
		::Eigen::TensorMap<::Eigen::Tensor<T,(RANK-CONTRACT_R)*2>> out(
			&got_data[0], expect.dimensions());

		if (ldensity == 1)
		{
			TensorMap<Tensor<T,RANK>> amap(adata.data(), atens.dimensions());
			auto b = this->make_sparse_src(btens, bdata, bindices);
			b.rContract(amap, contract_dims).write(out);
		}
		else if (rdensity == 1)
		{
			TensorMap<Tensor<T,RANK>> bmap(bdata.data(), btens.dimensions());
			auto a = this->make_sparse_src(atens, adata, aindices);
			a.contract(bmap, contract_dims).write(out);
		}
		else
		{
			auto a = this->make_sparse_src(atens, adata, aindices);
			auto b = this->make_sparse_src(btens, bdata, bindices);
			a.contract(b, contract_dims).write(out);
		}

		EXPECT_ARRCLOSE(expected_data, got_data, eps);
	}
};


struct CONTRACT_2D_DOUB : public ContractMutator<double,2> {};


struct CONTRACT_3D_DOUB : public ContractMutator<double,3> {};


static const double eps = std::numeric_limits<float>::epsilon();


TEST_F(CONTRACT_2D_DOUB, ContractSparseSparseToSparse_1ContractDims)
{
	generate_contract_to_sparse_testcase<1>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_2D_DOUB, ContractSparseSparseToSparse_2ContractDims)
{
	generate_contract_to_sparse_testcase<2>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractSparseSparseToSparse_1ContractDims)
{
	generate_contract_to_sparse_testcase<1>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractSparseSparseToSparse_2ContractDims)
{
	generate_contract_to_sparse_testcase<2>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractSparseSparseToSparse_3ContractDims)
{
	generate_contract_to_sparse_testcase<3>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_2D_DOUB, ContractSparseSparseToDense_1ContractDims)
{
	generate_contract_to_dense_testcase<1>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_2D_DOUB, ContractSparseSparseToDense_2ContractDims)
{
	generate_contract_to_dense_testcase<2>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractSparseSparseToDense_1ContractDims)
{
	generate_contract_to_dense_testcase<1>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractSparseSparseToDense_2ContractDims)
{
	generate_contract_to_dense_testcase<2>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractSparseSparseToDense_3ContractDims)
{
	generate_contract_to_dense_testcase<3>(0.5, 0.5, eps);
}


TEST_F(CONTRACT_2D_DOUB, ContractSparseDenseToDense_1ContractDims)
{
	generate_contract_to_dense_testcase<1>(0.5, 1, eps);
}


TEST_F(CONTRACT_2D_DOUB, ContractSparseDenseToDense_2ContractDims)
{
	generate_contract_to_dense_testcase<2>(0.5, 1, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractSparseDenseToDense_1ContractDims)
{
	generate_contract_to_dense_testcase<1>(0.5, 1, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractSparseDenseToDense_2ContractDims)
{
	generate_contract_to_dense_testcase<2>(0.5, 1, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractSparseDenseToDense_3ContractDims)
{
	generate_contract_to_dense_testcase<3>(0.5, 1, eps);
}


TEST_F(CONTRACT_2D_DOUB, ContractDenseSparseToDense_1ContractDims)
{
	generate_contract_to_dense_testcase<1>(1, 0.5, eps);
}


TEST_F(CONTRACT_2D_DOUB, ContractDenseSparseToDense_2ContractDims)
{
	generate_contract_to_dense_testcase<2>(1, 0.5, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractDenseSparseToDense_1ContractDims)
{
	generate_contract_to_dense_testcase<1>(1, 0.5, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractDenseSparseToDense_2ContractDims)
{
	generate_contract_to_dense_testcase<2>(1, 0.5, eps);
}


TEST_F(CONTRACT_3D_DOUB, ContractDenseSparseToDense_3ContractDims)
{
	generate_contract_to_dense_testcase<3>(1, 0.5, eps);
}
