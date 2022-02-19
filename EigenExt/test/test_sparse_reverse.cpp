#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, Reverse)
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

	// 8 9 0 6
	// 0 0 0 0
	// 7 0 0 5
	//
	// 0 7 1 0
	// 2 0 0 0
	// 0 0 3 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.reverse({true, false, false}).write(out0);
	std::vector<double> expect_data0 = { 8, 9, 6, 7, 5, 7, 1, 2, 3 };
	std::vector<size_t> expect_index0 = { 0, 1, 3, 8, 11, 13, 14, 16, 22 };
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);

	// 5 0 0 7
	// 0 0 0 0
	// 6 0 9 8
	//
	// 0 3 0 0
	// 0 0 0 2
	// 0 1 7 0
	nnz = 0;
	std::vector<double> data1;
	std::vector<size_t> index1;
	SparseTensorDst<double,std::vector<double>> out1(nnz, data1, index1, vector_info<double>());
	a.reverse({false, true, false}).write(out1);
	std::vector<double> expect_data1 = { 5, 7, 6, 9, 8, 3, 2, 1, 7 };
	std::vector<size_t> expect_index1 = { 0, 3, 8, 10, 11, 13, 19, 21, 22 };
	ASSERT_EQ(expect_index1.size(), nnz);
	EXPECT_ARREQ(expect_data1, data1);
	EXPECT_ARREQ(expect_index1, index1);

	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	//
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	nnz = 0;
	std::vector<double> data2;
	std::vector<size_t> index2;
	SparseTensorDst<double,std::vector<double>> out2(nnz, data2, index2, vector_info<double>());
	a.reverse({false, false, true}).write(out2);
	std::vector<double> expect_data2 = { 1, 7, 2, 3, 6, 9, 8, 5, 7  };
	std::vector<size_t> expect_index2 = { 1, 2, 7, 9, 12, 14, 15, 20, 23 };
	ASSERT_EQ(expect_index2.size(), nnz);
	EXPECT_ARREQ(expect_data2, data2);
	EXPECT_ARREQ(expect_index2, index2);
}


TEST(OPS, ReverseIdentity)
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
	a.reverse({false, false, false}).write(out0);
	ASSERT_EQ(indices.size(), nnz);
	EXPECT_ARREQ(data, data0);
	EXPECT_ARREQ(indices, index0);
}


TEST(OPS, ReverseShapeBasedIdentity)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	std::vector<double> data = { 6, 9, 8, 5, 7 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11 };

	SparseTensorSrc<double,3> a({4, 3, 1}, indices.size(), indices.data(), data.data());

	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.reverse({false, false, true}).write(out0);
	ASSERT_EQ(indices.size(), nnz);
	EXPECT_ARREQ(data, data0);
	EXPECT_ARREQ(indices, index0);
}
