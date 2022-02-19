
#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, ContractSparseSparseToSparse_1ContractDims_2D)
{
	ContractDimsT<uint8_t> dims = {{1,1}};

	std::vector<size_t> aindices = {2,4};
	std::vector<double> adata = {1,2};
	SparseTensorSrc<double,2> a({6,1}, aindices.size(), aindices.data(), adata.data());

	std::vector<size_t> bindices = {1,4};
	std::vector<double> bdata = {3,5};
	SparseTensorSrc<double,2> b({5,1}, bindices.size(), bindices.data(), bdata.data());

	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.contract(b, dims).write(out0);

	std::vector<double> expect_data0 = {
		0,0,0,0,0,
		0,0,0,0,0,
		0,3,0,0,5,
		0,0,0,0,0,
		0,6,0,0,10,
		0,0,0,0,0
	};
	std::vector<double> result(expect_data0.size(), 0);
	for (size_t i = 0; i < nnz; ++i)
	{
		result[index0[i]] = data0[i];
	}
	EXPECT_ARREQ(expect_data0, result);
}


TEST(OPS, ContractSparseSparseToSparse_1ContractDims_3D)
{
	ContractDimsT<uint8_t> dims = {{2,2}};

	std::vector<size_t> aindices = {2,3,5,6,9,11,13,14,16,19,20,22,24,25,26,34};
	std::vector<double> adata = {1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47};
	SparseTensorSrc<double,3> a({3,3,4}, aindices.size(), aindices.data(), adata.data());

	std::vector<size_t> bindices = {2,3,5,6,7,8,9,11,13,14,16,17,20,25,26,27};
	std::vector<double> bdata = {53,59,61,67,71,73,79,83,97,101,103,107,109,113,127,131};
	SparseTensorSrc<double,3> b({1,7,4}, bindices.size(), bindices.data(), bdata.data());

	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.contract(b, dims).write(out0);

	std::vector<double> expect_data0 = {
		497,511,553,0,581,0,679,
		2323,0,2369,2461,0,0,2507,
		3710,803,3909,3162,913,61,4295,
		0,0,106,118,0,122,134,
		4054,949,4220,3317,1079,0,4640,
		1207,1241,1502,177,1411,183,1850,
		3737,0,4076,4254,0,305,4368,
		5490,1387,5724,4387,6888,5969,12469,
		4343,0,4429,4601,0,0,4687,
	};
	std::vector<double> result(expect_data0.size(), 0);
	for (size_t i = 0; i < nnz; ++i)
	{
		result[index0[i]] = data0[i];
	}
	EXPECT_ARREQ(expect_data0, result);
}


TEST(OPS, ContractSparseSparseToSparse_2ContractDims)
{
	ContractDimsT<uint8_t> dims = {{2,2}, {1,1}};

	std::vector<size_t> aindices = {2,4,6,11,12,14,16,18,19,21,23,24,25,26,30,31,33,34,36,37,39,40,41,43,44,45,46,49,50,51,53,54,56,59};
	std::vector<double> adata = {
		2,3,5,7,11,13,17,19,23,29,
		31,37,41,43,47,53,59,61,67,71,
		73,79,83,89,97,101,103,107,109,113,
		127,131,137,139
	};
	SparseTensorSrc<double,3> a({3,4,5}, aindices.size(), aindices.data(), adata.data());

	std::vector<size_t> bindices = {0,3,4,5,6,8,9,12,13,14,20,22,23,26,27,28,34,36,37,38,39,40,41,44,45,46,49,50,53,55,56,58,59,60,61,64,67,68,73,75,76,79};
	std::vector<double> bdata = {
		149,151,157,163,167,173,179,181,191,193,
		197,199,211,223,227,229,233,239,241,251,
		257,263,269,271,277,281,283,293,307,311,
		313,317,331,337,347,349,353,359,367,373,
		379,383
	};
	SparseTensorSrc<double,3> b({4,4,5}, bindices.size(), bindices.data(), bdata.data());

	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.contract(b, dims).write(out0);

	std::vector<double> expect_data0 = {110460,154377,49068,75879,134201,111730,84723,100607,175340,77097,42119,201037};
	std::vector<double> result(expect_data0.size(), 0);
	for (size_t i = 0; i < nnz; ++i)
	{
		result[index0[i]] = data0[i];
	}

	EXPECT_ARREQ(expect_data0, result);
}


TEST(OPS, ContractSparseSparseToSparse)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> adata = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> aindices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };
	SparseTensorSrc<double,3> a({4, 3, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 1 9 0
	// 0 0 1 0
	// 2 0 0 0
	// 2 0 4 0
	std::vector<double> bdata = { 1, 9, 1, 2, 2, 4 };
	std::vector<size_t> bindices = { 1, 2, 6, 8, 12, 14 };
	SparseTensorSrc<double,3> b({4, 4, 1}, bindices.size(), bindices.data(), bdata.data());

	// 34  6 86  0
	//  0  0  0  0
	// 14  5 73  0
	//
	// 14  0  1  0
	//  4  0  8  0
	//  0  0  3  0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.contract(b, ContractDimsT<uint8_t>{{0, 1}}).write(out0);
	std::vector<double> expect_data0 = {
		34, 6, 86,
		14, 5, 73,
		14, 1,
		4, 8, 3
	};
	std::vector<size_t> expect_index0 = {
		0, 1, 2,
		8, 9, 10,
		12, 14,
		16, 18, 22,
	};
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, ContractSparseSparseToDense)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> adata = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> aindices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };
	SparseTensorSrc<double,3> a({4, 3, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 1 9 0
	// 0 0 1 0
	// 2 0 0 0
	// 2 0 4 0
	std::vector<double> bdata = { 1, 9, 1, 2, 2, 4 };
	std::vector<size_t> bindices = { 1, 2, 6, 8, 12, 14 };
	SparseTensorSrc<double,3> b({4, 4, 1}, bindices.size(), bindices.data(), bdata.data());

	// 34  6 86  0
	//  0  0  0  0
	// 14  5 73  0
	//
	// 14  0  1  0
	//  4  0  8  0
	//  0  0  3  0
	std::vector<double> data0(24);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	a.contract(b, ContractDimsT<uint8_t>{{0, 1}}).write(out0);
	std::vector<double> expect_data0 = {
		34, 6, 86, 0,
		0, 0, 0, 0,
		14, 5, 73, 0,
		14, 0, 1, 0,
		4, 0, 8, 0,
		0, 0, 3, 0
	};
	EXPECT_ARREQ(expect_data0, data0);
}


TEST(OPS, ContractSparseSparseAppearDenseToDense)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> adata = {
		6, 0, 9, 8,
		0, 0, 0, 0,
		5, 0, 0, 7,
		0, 1, 7, 0,
		0, 0, 0, 2,
		0, 3, 0, 0
	};
	std::vector<size_t> aindices(adata.size());
	std::iota(aindices.begin(), aindices.end(), 0);
	SparseTensorSrc<double,3> a({4, 3, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 1 9 0
	// 0 0 1 0
	// 2 0 0 0
	// 2 0 4 0
	std::vector<double> bdata = {
		0, 1, 9, 0,
		0, 0, 1, 0,
		2, 0, 0, 0,
		2, 0, 4, 0
	};
	std::vector<size_t> bindices(bdata.size());
	std::iota(bindices.begin(), bindices.end(), 0);
	SparseTensorSrc<double,3> b({4, 4, 1}, bindices.size(), bindices.data(), bdata.data());

	// 34  6 86  0
	//  0  0  0  0
	// 14  5 73  0
	//
	// 14  0  1  0
	//  4  0  8  0
	//  0  0  3  0
	std::vector<double> data0(24);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	a.contract(b, ContractDimsT<uint8_t>{{0, 1}}).write(out0);
	std::vector<double> expect_data0 = {
		34, 6, 86, 0,
		0, 0, 0, 0,
		14, 5, 73, 0,
		14, 0, 1, 0,
		4, 0, 8, 0,
		0, 0, 3, 0
	};
	EXPECT_ARREQ(expect_data0, data0);
}


TEST(OPS, ContractSparseDenseToDense_1D_2ContractDims)
{
	std::vector<size_t> left_indices = {0,2};
	std::vector<double> left_data = {
		7,11
	};
	SparseTensorSrc<double,2> a({4,1}, left_indices.size(), left_indices.data(), left_data.data());

	std::vector<double> right_data = {
		1,2,3,5
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,2>> b(right_data.data(), {4, 1});

	double data0;
	::Eigen::TensorMap<::Eigen::Tensor<double,0>> out0(&data0);
	a.contract(b, ContractDimsT<uint8_t>{{1, 1}, {0, 0}}).write(out0);

	double expected_data = 7*1+11*3;
	EXPECT_DOUBLE_EQ(expected_data, data0);
}


TEST(OPS, ContractSparseDenseToDense_2D_1ContractDims)
{
	std::vector<size_t> left_indices = {
		0,
		3,
		6,7,8,
		11
	};
	std::vector<double> left_data = {
		5,
		7,
		11,13,17,
		19
	};
	SparseTensorSrc<double,2> a({3,4}, left_indices.size(), left_indices.data(), left_data.data());

	std::vector<double> right_data = {
		1,2,3
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,2>> b(right_data.data(), {3, 1});

	std::vector<double> data0(4);
	::Eigen::TensorMap<::Eigen::Tensor<double,2>> out0(data0.data(), {1, 4});
	a.contract(b, ContractDimsT<uint8_t>{{0, 0}}).write(out0);

	std::vector<double> expected_data = {
		1*5,
		1*7,
		1*11+2*13+3*17,
		3*19
	};
	EXPECT_ARREQ(expected_data, data0);
}


TEST(OPS, ContractSparseDenseToDense_2D_2ContractDims)
{
	std::vector<size_t> left_indices = {0,3,5,6};
	std::vector<double> left_data = {
		1,2,3,5
	};
	SparseTensorSrc<double,2> a({4,2}, left_indices.size(), left_indices.data(), left_data.data());

	std::vector<double> right_data = {
		7,11,13,17,
		19,21,23,29
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,2>> b(right_data.data(), {4, 2});

	double data0;
	::Eigen::TensorMap<::Eigen::Tensor<double,0>> out0(&data0);
	a.contract(b, ContractDimsT<uint8_t>{{1, 1}, {0, 0}}).write(out0);

	double expected_data = 1*7+2*17+3*21+5*23;
	EXPECT_DOUBLE_EQ(expected_data, data0);
}


TEST(OPS, ContractSparseDenseToDense)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> adata = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> aindices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };
	SparseTensorSrc<double,3> a({4, 3, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 1 9 0
	// 0 0 1 0
	// 2 0 0 0
	// 2 0 4 0
	std::vector<double> bdata = {
		0, 1, 9, 0,
		0, 0, 1, 0,
		2, 0, 0, 0,
		2, 0, 4, 0,
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> b(bdata.data(), {4, 4, 1});

	// 34  6 86  0
	//  0  0  0  0
	// 14  5 73  0
	//
	// 14  0  1  0
	//  4  0  8  0
	//  0  0  3  0
	std::vector<double> data0(24);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	a.contract(b, ContractDimsT<uint8_t>{{0, 1}}).write(out0);
	std::vector<double> expect_data0 = {
		34, 6, 86, 0,
		0, 0, 0, 0,
		14, 5, 73, 0,
		14, 0, 1, 0,
		4, 0, 8, 0,
		0, 0, 3, 0
	};
	EXPECT_ARREQ(expect_data0, data0);
}


TEST(OPS, ContractDenseSparseToDense_1D_1ContractDims)
{
	std::vector<double> left_data = {
		1, 2
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,2>> a(left_data.data(), {1,2});

	std::vector<size_t> right_indices = {0,2,5};
	std::vector<double> right_data = {
		3, 5, 7
	};
	SparseTensorSrc<double,2> b({1,6},
		right_indices.size(), right_indices.data(), right_data.data());

	std::vector<double> data0(12);
	::Eigen::TensorMap<::Eigen::Tensor<double,2>> out0(data0.data(), {6,2});
	b.rContract(a, ContractDimsT<uint8_t>{{0, 0}}).write(out0);

	std::vector<double> expected_data = {
		1*3, 0, 1*5, 0, 0, 1*7,
		2*3, 0, 2*5, 0, 0, 2*7
	};
	EXPECT_ARREQ(expected_data, data0);
}


TEST(OPS, ContractDenseSparseToDense_1D_2ContractDims)
{
	std::vector<double> left_data = {3,5,7,11,13};
	::Eigen::TensorMap<::Eigen::Tensor<double,2>> a(left_data.data(), {5,1});

	std::vector<size_t> right_indices = {0,2};
	std::vector<double> right_data = {1,2};
	SparseTensorSrc<double,2> b({5,1}, right_indices.size(), right_indices.data(), right_data.data());

	double data0;
	::Eigen::TensorMap<::Eigen::Tensor<double,0>> out0(&data0);
	b.rContract(a, ContractDimsT<uint8_t>{{1, 1}, {0, 0}}).write(out0);

	double expected_data = 1*3 + 2*7;
	EXPECT_DOUBLE_EQ(expected_data, data0);
}


TEST(OPS, ContractDenseSparseToDense_2D_2ContractDims)
{
	std::vector<double> left_data = {5,7,11,13,17,19};
	::Eigen::TensorMap<::Eigen::Tensor<double,2>> a(left_data.data(), {2,3});

	std::vector<size_t> right_indices = {0,3,4};
	std::vector<double> right_data = {1,2,3};
	SparseTensorSrc<double,2> b({2,3}, right_indices.size(), right_indices.data(), right_data.data());

	double data0;
	::Eigen::TensorMap<::Eigen::Tensor<double,0>> out0(&data0);
	b.rContract(a, ContractDimsT<uint8_t>{{1, 1}, {0, 0}}).write(out0);

	double expected_data = 1*5 + 2*13 + 3*17;
	EXPECT_DOUBLE_EQ(expected_data, data0);
}


TEST(OPS, ContractDenseSparseToDense)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> adata = {
		6, 0, 9, 8,
		0, 0, 0, 0,
		5, 0, 0, 7,

		0, 1, 7, 0,
		0, 0, 0, 2,
		0, 3, 0, 0,
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> a(adata.data(), {4, 3, 2});

	// 0 1 9 0
	// 0 0 1 0
	// 2 0 0 0
	// 2 0 4 0
	std::vector<double> bdata = { 1, 9, 1, 2, 2, 4 };
	std::vector<size_t> bindices = { 1, 2, 6, 8, 12, 14 };
	SparseTensorSrc<double,3> b({4, 4, 1}, bindices.size(), bindices.data(), bdata.data());

	// 34  6 86  0
	//  0  0  0  0
	// 14  5 73  0
	//
	// 14  0  1  0
	//  4  0  8  0
	//  0  0  3  0
	std::vector<double> data0(24);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	b.rContract(a, ContractDimsT<uint8_t>{{0, 1}}).write(out0);
	std::vector<double> expect_data0 = {
		34, 6, 86, 0,
		0, 0, 0, 0,
		14, 5, 73, 0,

		14, 0, 1, 0,
		4, 0, 8, 0,
		0, 0, 3, 0
	};
	EXPECT_ARREQ(expect_data0, data0);
}
