#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, Broadcast)
{
	// 6 9
	// 0 0
	// 5 0
	std::vector<double> data = { 6, 9, 5 };
	std::vector<size_t> indices = { 0, 1, 4 };

	SparseTensorSrc<double,3> a({2, 3, 1}, indices.size(), indices.data(), data.data());

	// 6 9 6 9
	// 0 0 0 0
	// 5 0 5 0
	//
	// 6 9 6 9
	// 0 0 0 0
	// 5 0 5 0
	//
	// 6 9 6 9
	// 0 0 0 0
	// 5 0 5 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.template broadcast<size_t>({2, 1, 3}).write(out0);
	std::vector<double> expect_data0 = {
		6, 9, 6, 9, 5, 5,
		6, 9, 6, 9, 5, 5,
		6, 9, 6, 9, 5, 5
	};
	std::vector<size_t> expect_index0 = {
		0, 1, 2, 3, 8, 10,
		12, 13, 14, 15, 20, 22,
		24, 25, 26, 27, 32, 34,
	};
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, RedundantBroadcast)
{
	// 6 9
	// 0 0
	// 5 0
	std::vector<double> data = { 6, 9, 5 };
	std::vector<size_t> indices = { 0, 1, 4 };

	SparseTensorSrc<double,3> a({2, 3, 1}, indices.size(), indices.data(), data.data());

	// 6 9 6 9
	// 0 0 0 0
	// 5 0 5 0
	//
	// 6 9 6 9
	// 0 0 0 0
	// 5 0 5 0
	//
	// 6 9 6 9
	// 0 0 0 0
	// 5 0 5 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.template broadcast<size_t>({1, 1, 1}).write(out0);
	ASSERT_EQ(indices.size(), nnz);
	EXPECT_ARREQ(data, data0);
	EXPECT_ARREQ(indices, index0);
}
