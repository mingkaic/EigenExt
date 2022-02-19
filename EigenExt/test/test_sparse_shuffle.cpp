#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, Shuffle)
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

	// 6 0
	// 0 0
	// 5 0
	//
	// 0 1
	// 0 0
	// 0 3
	//
	// 9 7
	// 0 0
	// 0 0
	//
	// 8 0
	// 0 2
	// 7 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.shuffle(std::array<size_t,3>{2, 1, 0}).write(out0);
	std::vector<double> expect_data0 = { 6, 5, 1, 3, 9, 7, 8, 2, 7 };
	std::vector<size_t> expect_index0 = { 0, 4, 7, 11, 12, 13, 18, 21, 22 };
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);

	// 6 0 5
	// 0 0 0
	//
	// 0 0 0
	// 1 0 3
	//
	// 9 0 0
	// 7 0 0
	//
	// 8 0 7
	// 0 2 0
	nnz = 0;
	std::vector<double> data1;
	std::vector<size_t> index1;
	SparseTensorDst<double,std::vector<double>> out1(nnz, data1, index1, vector_info<double>());
	a.shuffle(std::array<size_t,3>{1, 2, 0}).write(out1);
	std::vector<double> expect_data1 = { 6, 5, 1, 3, 9, 7, 8, 7, 2 };
	std::vector<size_t> expect_index1 = { 0, 2, 9, 11, 12, 15, 18, 20, 22 };
	ASSERT_EQ(expect_index1.size(), nnz);
	EXPECT_ARREQ(expect_data1, data1);
	EXPECT_ARREQ(expect_index1, index1);

	// 6 0 5
	// 0 0 0
	// 9 0 0
	// 8 0 7
	//
	// 0 0 0
	// 1 0 3
	// 7 0 0
	// 0 2 0
	nnz = 0;
	std::vector<double> data2;
	std::vector<size_t> index2;
	SparseTensorDst<double,std::vector<double>> out2(nnz, data2, index2, vector_info<double>());
	a.shuffle(std::array<size_t,3>{1, 0, 2}).write(out2);
	std::vector<double> expect_data2 = { 6, 5, 9, 8, 7, 1, 3, 7, 2 };
	std::vector<size_t> expect_index2 = { 0, 2, 6, 9, 11, 15, 17, 18, 22 };
	ASSERT_EQ(expect_index2.size(), nnz);
	EXPECT_ARREQ(expect_data2, data2);
	EXPECT_ARREQ(expect_index2, index2);
}


TEST(OPS, ShuffleIdentity)
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
	a.shuffle(std::array<size_t,3>{0, 1, 2}).write(out0);
	ASSERT_EQ(indices.size(), nnz);
	EXPECT_ARREQ(data, data0);
	EXPECT_ARREQ(indices, index0);
}
