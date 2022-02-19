
#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, ConvolveSparseSparseToSparse_KernIdentity)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	// 0 1 0 2
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	// 1 0 2 0
	std::vector<double> adata = {
		6, 9, 8,
		5, 7,
		1, 2,
		1, 7,
		2, 3,
		1, 2
	};
	std::vector<size_t> aindices = {
		0, 2, 3,
		8, 11,
		13, 15,
		17, 18,
		23, 25,
		28, 30
	};

	SparseTensorSrc<double,3> a({4, 4, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 0 2
	// 1 0 0
	//
	// 9 1 0
	// 0 0 0
	std::vector<double> bdata = { 2, 1, 9, 1 };
	std::vector<size_t> bindices = { 2, 3, 6, 7 };

	SparseTensorSrc<double,3> b({3, 2, 2}, bindices.size(), bindices.data(), bdata.data());

	// 19 32
	// 5  0
	// 3  42
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.convolve(b, std::array<int,3>{0, 1, 2}).write(out0);
	std::vector<double> expect_data0 = {
		19, 32, 5, 3, 42
	};
	std::vector<size_t> expect_index0 = {
		0, 1, 2, 4, 5
	};
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_VECEQ(expect_data0, data0);
	EXPECT_VECEQ(expect_index0, index0);
}


TEST(OPS, ConvolveSparseSparseToSparse)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	// 0 1 0 2
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	// 1 0 2 0
	std::vector<double> adata = {
		6, 9, 8,
		5, 7,
		1, 2,
		1, 7,
		2, 3,
		1, 2
	};
	std::vector<size_t> aindices = {
		0, 2, 3,
		8, 11,
		13, 15,
		17, 18,
		23, 25,
		28, 30
	};

	SparseTensorSrc<double,3> a({4, 4, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 1
	// 9 0
	//
	// 0 0
	// 1 0
	//
	// 2 0
	// 0 0
	std::vector<double> bdata = { 1, 9, 1, 2 };
	std::vector<size_t> bindices = { 1, 2, 6, 8 };

	SparseTensorSrc<double,3> b({2, 2, 3}, bindices.size(), bindices.data(), bdata.data());

	// kernel is treated as
	// 0 0 2
	// 1 0 0
	//
	// 9 1 0
	// 0 0 0

	// * 1
	// 2 *
	//
	// * *
	// 6 *
	//
	// 8 *
	// * *

	// * * 8
	// 1 * *
	//
	// 2 6 *
	// * * *

	// 0  *  2     *  2  3
	// *  *  *     *  *  *
	// *  17 18    17 18 *
	// *  *  *     *  *  23
	//
	// *  *  *     *  *  *
	// 8  *  *     *  *  11
	// *  *  *     *  *  23
	// *  25 *     25 *  *
	//
	// 8  *  *     *  *  11
	// *  13 *     13 *  15
	// *  25 *     25 *  *
	// 28 *  30    *  30 *

	// 2x8,17x6->0
	// 3x8,17x2,18x6->1
	// 8x1->2
	// 25x6->4<
	// 11x8,13x1,25x2->5


	// 19 32
	// 5  0
	// 3  42
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.convolve(b, std::array<int,3>{1, 2, 0}).write(out0);
	std::vector<double> expect_data0 = {
		19, 32, 5, 3, 42
	};
	std::vector<size_t> expect_index0 = {
		0, 1, 2, 4, 5
	};
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_VECEQ(expect_data0, data0);
	EXPECT_VECEQ(expect_index0, index0);
}


TEST(OPS, ConvolveSparseSparseToDense)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	// 0 1 0 2
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	// 1 0 2 0
	std::vector<double> adata = {
		6, 9, 8,
		5, 7,
		1, 2,
		1, 7,
		2, 3,
		1, 2
	};
	std::vector<size_t> aindices = {
		0, 2, 3,
		8, 11,
		13, 15,
		17, 18,
		23, 25,
		28, 30
	};

	SparseTensorSrc<double,3> a({4, 4, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 1
	// 9 0
	//
	// 0 0
	// 1 0
	//
	// 2 0
	// 0 0
	std::vector<double> bdata = { 1, 9, 1, 2 };
	std::vector<size_t> bindices = { 1, 2, 6, 8 };

	SparseTensorSrc<double,3> b({2, 2, 3}, bindices.size(), bindices.data(), bdata.data());

	// kernel is treated as
	// 0 0 2
	// 1 0 0
	//
	// 9 1 0
	// 0 0 0

	// 19 32
	// 5  0
	// 3  42
	std::vector<double> data0(6);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {2, 3, 1});
	a.convolve(b, std::array<int,3>{1, 2, 0}).write(out0);
	std::vector<double> expect_data0 = {
		19, 32, 5, 0, 3, 42
	};
	EXPECT_VECEQ(expect_data0, data0);
}


TEST(OPS, ConvolveSparseDenseToSparse_KernIdentity)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	// 0 1 0 2
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	// 1 0 2 0
	std::vector<double> adata = {
		6, 9, 8,
		5, 7,
		1, 2,
		1, 7,
		2, 3,
		1, 2
	};
	std::vector<size_t> aindices = {
		0, 2, 3,
		8, 11,
		13, 15,
		17, 18,
		23, 25,
		28, 30
	};

	SparseTensorSrc<double,3> a({4, 4, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 0 2
	// 1 0 0
	//
	// 9 1 0
	// 0 0 0
	std::vector<double> bdata = {
		0, 0, 2,
		1, 0, 0,

		9, 1, 0,
		0, 0, 0
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> b(bdata.data(), {3, 2, 2});

	// 19 32
	// 5  0
	// 3  42
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.convolve(b, std::array<int,3>{0, 1, 2}).write(out0);
	std::vector<double> expect_data0 = {
		19, 32, 5, 3, 42
	};
	std::vector<size_t> expect_index0 = {
		0, 1, 2, 4, 5
	};
	EXPECT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, ConvolveSparseDenseToSparse)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	// 0 1 0 2
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	// 1 0 2 0
	std::vector<double> adata = {
		6, 9, 8,
		5, 7,
		1, 2,
		1, 7,
		2, 3,
		1, 2
	};
	std::vector<size_t> aindices = {
		0, 2, 3,
		8, 11,
		13, 15,
		17, 18,
		23, 25,
		28, 30
	};

	SparseTensorSrc<double,3> a({4, 4, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 1
	// 9 0
	//
	// 0 0
	// 1 0
	//
	// 2 0
	// 0 0
	std::vector<double> bdata = {
		0, 1,
		9, 0,

		0, 0,
		1, 0,

		2, 0,
		0, 0
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> b(bdata.data(), {2, 2, 3});

	// kernel is treated as
	// 0 0 2
	// 1 0 0
	//
	// 9 1 0
	// 0 0 0

	// 19 32
	// 5  0
	// 3  42
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.convolve(b, std::array<int,3>{1, 2, 0}).write(out0);
	std::vector<double> expect_data0 = {
		19, 32, 5, 3, 42
	};
	std::vector<size_t> expect_index0 = {
		0, 1, 2, 4, 5
	};
	EXPECT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, ConvolveSparseDenseToDense)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	// 0 1 0 2
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	// 1 0 2 0
	std::vector<double> adata = {
		6, 9, 8,
		5, 7,
		1, 2,
		1, 7,
		2, 3,
		1, 2
	};
	std::vector<size_t> aindices = {
		0, 2, 3,
		8, 11,
		13, 15,
		17, 18,
		23, 25,
		28, 30
	};

	SparseTensorSrc<double,3> a({4, 4, 2}, aindices.size(), aindices.data(), adata.data());

	// 0 1
	// 9 0
	//
	// 0 0
	// 1 0
	//
	// 2 0
	// 0 0
	std::vector<double> bdata = {
		0, 1,
		9, 0,

		0, 0,
		1, 0,

		2, 0,
		0, 0
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> b(bdata.data(), {2, 2, 3});

	// kernel is treated as
	// 0 0 2
	// 1 0 0
	//
	// 9 1 0
	// 0 0 0

	// 19 32
	// 5  0
	// 3  42
	std::vector<double> data0(6);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {2, 3, 1});
	a.convolve(b, std::array<int,3>{1, 2, 0}).write(out0);
	std::vector<double> expect_data0 = {
		19, 32, 5, 0, 3, 42
	};
	EXPECT_VECEQ(expect_data0, data0);
}


TEST(OPS, ConvolveDenseSparseToDense_KernIdentity)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	// 0 1 0 2
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	// 1 0 2 0
	std::vector<double> adata = {
		6, 0, 9, 8,
		0, 0, 0, 0,
		5, 0, 0, 7,
		0, 1, 0, 2,
		0, 1, 7, 0,
		0, 0, 0, 2,
		0, 3, 0, 0,
		1, 0, 2, 0
	};

	::Eigen::TensorMap<::Eigen::Tensor<double,3>> a(adata.data(), {4, 4, 2});

	// 0 0 2
	// 1 0 0
	//
	// 9 1 0
	// 0 0 0
	std::vector<double> bdata = { 2, 1, 9, 1 };
	std::vector<size_t> bindices = { 2, 3, 6, 7 };

	SparseTensorSrc<double,3> b({3, 2, 2}, bindices.size(), bindices.data(), bdata.data());

	// 19 32
	// 5  0
	// 3  42
	std::vector<double> data0(6);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {2, 3, 1});
	b.rConvolve(a, std::array<int,3>{0, 1, 2}).write(out0);
	std::vector<double> expect_data0 = {
		19, 32, 5, 0, 3, 42
	};
	EXPECT_VECEQ(expect_data0, data0);
}


TEST(OPS, ConvolveDenseSparseToDense)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	// 0 1 0 2
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	// 1 0 2 0
	std::vector<double> adata = {
		6, 0, 9, 8,
		0, 0, 0, 0,
		5, 0, 0, 7,
		0, 1, 0, 2,
		0, 1, 7, 0,
		0, 0, 0, 2,
		0, 3, 0, 0,
		1, 0, 2, 0
	};

	::Eigen::TensorMap<::Eigen::Tensor<double,3>> a(adata.data(), {4, 4, 2});

	// 0 1
	// 9 0
	//
	// 0 0
	// 1 0
	//
	// 2 0
	// 0 0
	std::vector<double> bdata = { 1, 9, 1, 2 };
	std::vector<size_t> bindices = { 1, 2, 6, 8 };

	SparseTensorSrc<double,3> b({2, 2, 3}, bindices.size(), bindices.data(), bdata.data());

	// kernel is treated as
	// 0 0 2
	// 1 0 0
	//
	// 9 1 0
	// 0 0 0

	// 19 32
	// 5  0
	// 3  42
	std::vector<double> data0(6);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {2, 3, 1});
	b.rConvolve(a, std::array<int,3>{1, 2, 0}).write(out0);
	std::vector<double> expect_data0 = {
		19, 32, 5, 0, 3, 42
	};
	EXPECT_VECEQ(expect_data0, data0);
}
