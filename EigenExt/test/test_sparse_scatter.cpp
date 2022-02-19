#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, Scatter)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> data = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };

	SparseTensorSrc<double,3> a({4, 3, 2}, indices.size(), indices.data(), data.data());

	// 6 0 0 0 9 0 8 0
	// 0 0 0 0 0 0 0 0
	// 5 0 0 0 0 0 7 0
	//
	// 0 0 0 0 0 0 0 0
	// 0 0 0 0 0 0 0 0
	// 0 0 0 0 0 0 0 0
	//
	// 0 0 1 0 7 0 0 0
	// 0 0 0 0 0 0 2 0
	// 0 0 3 0 0 0 0 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.template scatter<size_t>({2, 1, 2}, {8, 3, 3}).write(out0);
	std::vector<double> expect_data0 = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> expect_index0 = { 0, 4, 6, 16, 22, 50, 52, 62, 66 };
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, ScatterReshape)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> data = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };

	SparseTensorSrc<double,3> a({4, 3, 2}, indices.size(), indices.data(), data.data());

	// 6 0 9 8 0 0 0 0
	// 0 0 0 0 0 0 0 0
	// 5 0 0 7 0 0 0 0
	// 0 0 0 0 0 0 0 0
	//
	// 0 1 7 0 0 0 0 0
	// 0 0 0 2 0 0 0 0
	// 0 3 0 0 0 0 0 0
	// 0 0 0 0 0 0 0 0
	//
	// 0 0 0 0 0 0 0 0
	// 0 0 0 0 0 0 0 0
	// 0 0 0 0 0 0 0 0
	// 0 0 0 0 0 0 0 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.template scatter<size_t>({1, 1, 1}, {8, 4, 3}).write(out0);
	std::vector<double> expect_data0 = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> expect_index0 = {
		0, 2, 3,
		16, 19,
		33, 34,
		43, 49
	};
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, ScatterIdentity)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> data = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };

	SparseTensorSrc<double,3> a({4, 3, 2}, indices.size(), indices.data(), data.data());

	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.template scatter<size_t>({1, 1, 1}, {4, 3, 2}).write(out0);
	ASSERT_EQ(indices.size(), nnz);
	EXPECT_ARREQ(data, data0);
	EXPECT_ARREQ(indices, index0);
}
