#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(BUFFER, Assign)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	AccumulateBuffer<double,size_t> buff;
	buff.idx2val_ = {
		{0, 6},
		{2, 9},
		{3, 8},
		{8, 5},
		{11, 7},
		{13, 1},
		{14, 7},
		{19, 2},
		{21, 3}
	};

	std::vector<double> dst_data(24);
	std::iota(dst_data.begin(), dst_data.end(), 0);
	TensorMap<Tensor<double,3>> dst(dst_data.data(), {4, 3, 2});
	buff.assign_nonzeros(dst, [](const double& l, const double& r) { return r; });

	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	buff.write(out0);

	std::vector<size_t> expect_indices = { 2, 3, 8, 11, 13, 14, 19, 21 };
	std::vector<double> expect_data(expect_indices.begin(), expect_indices.end());
	ASSERT_EQ(expect_indices.size(), nnz);
	EXPECT_ARREQ(expect_data, data0);
	EXPECT_ARREQ(expect_indices, index0);
}


TEST(BUFFER, Unary)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	AccumulateBuffer<double,size_t> buff;
	buff.idx2val_ = {
		{0, 6},
		{2, 9},
		{3, 8},
		{8, 5},
		{11, 7},
		{13, 1},
		{14, 7},
		{19, 2},
		{21, 3}
	};

	buff.unaryExpr([](const double& e) { return -e; });

	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	buff.write(out0);

	std::vector<double> expect_data = { -6, -9, -8, -5, -7, -1, -7, -2, -3 };
	std::vector<size_t> expect_indices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };
	ASSERT_EQ(expect_indices.size(), nnz);
	EXPECT_ARREQ(expect_data, data0);
	EXPECT_ARREQ(expect_indices, index0);
}
