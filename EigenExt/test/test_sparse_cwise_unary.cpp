
#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, CwiseNegToSparse)
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

	// -6 0 -9 -8
	// 0 0 0 0
	// -5 0 0 -7
	//
	// 0 -1 -7 0
	// 0 0 0 -2
	// 0 -3 0 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.cwiseUnary(::Eigen::internal::scalar_opposite_op<double>()).write(out0);
	std::vector<double> expect_data0 = { -6, -9, -8, -5, -7, -1, -7, -2, -3 };
	std::vector<size_t> expect_index0 = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, CwiseNegToDense)
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

	// -6 0 -9 -8
	// 0 0 0 0
	// -5 0 0 -7
	//
	// 0 -1 -7 0
	// 0 0 0 -2
	// 0 -3 0 0
	std::vector<double> data0(24);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	a.cwiseUnary(::Eigen::internal::scalar_opposite_op<double>()).write(out0);
	std::vector<double> expect_data = {
		-6, 0 , -9, -8,
		0 , 0 , 0 , 0,
		-5, 0 , 0 , -7,
		0 , -1, -7, 0,
		0 , 0 , 0 , -2,
		0 , -3, 0 , 0
	};
	EXPECT_ARREQ(expect_data, data0);
}


TEST(OPS, CwiseThreshold)
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

	// 6/3 0 9/3 8/3
	// 0 0 0 0
	// 5/3 0 0 7/3
	//
	// 0 0 7/3 0
	// 0 0 0 2/3
	// 0 3/3 0 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.cwiseUnary([](const double& a) { return a / 3; }, 0.5).write(out0);
	std::vector<double> expect_data0 = {
		2, 3, (double)8/3,
		(double)5/3, (double)7/3,
		(double)7/3, (double)2/3, 1
	};
	std::vector<size_t> expect_index0 = {
		0, 2, 3,
		8, 11,
		14, 19, 21
	};
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, NegativeCwiseCos)
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

	// 2 1 2 2
	// 1 1 1 1
	// 2 1 1 2
	//
	// 1 2 2 1
	// 1 1 1 2
	// 1 2 1 1
	std::vector<double> data0(24, 2);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	a.cwiseUnary(::Eigen::internal::scalar_cos_op<double>()).negative_write(out0);
	std::vector<double> expect_data = {
		2, 1, 2, 2,
		1, 1, 1, 1,
		2, 1, 1, 2,
		1, 2, 2, 1,
		1, 1, 1, 2,
		1, 2, 1, 1
	};
	EXPECT_ARREQ(expect_data, data0);
}
