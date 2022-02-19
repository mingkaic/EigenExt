#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, ChipToSrc)
{
	// 6 0
	// 0 0
	// 5 0
	//
	// 0 1
	// 0 0
	// 0 3
	std::vector<double> data0 = { 6, 5, 1, 3 };
	std::vector<size_t> indices0 = { 0, 4, 7, 11 };

	SparseTensorSrc<double,3> a({2, 3, 2}, indices0.size(), indices0.data(), data0.data());

	// 9 8
	// 0 0
	// 0 7
	//
	// 7 0
	// 0 2
	// 0 0
	std::vector<double> data1 = { 9, 8, 7, 7, 2 };
	std::vector<size_t> indices1 = { 0, 1, 5, 6, 9 };

	SparseTensorSrc<double,3> b({2, 3, 2}, indices1.size(), indices1.data(), data1.data());

	// 7 0 0
	// 0 3 1
	// 0 1 0
	//
	// 0 0 0
	// 3 0 0
	// 0 0 0
	std::vector<double> data2 = { 7, 3, 1, 1, 3 };
	std::vector<size_t> indices2 = { 0, 4, 5, 7, 12 };

	SparseTensorSrc<double,3> c({3, 3, 2}, indices2.size(), indices2.data(), data2.data());

	// 6 0 9 8 7 0 0
	// 0 0 0 0 0 3 1
	// 5 0 0 7 0 1 0
	//
	// 0 1 7 0 0 0 0
	// 0 0 0 2 3 0 0
	// 0 3 0 0 0 0 0
	size_t nnz = 0;
	std::vector<double> data4;
	std::vector<size_t> index4;
	ShapeT<3> outdim = {7, 3, 2};
	SparseTensorDst<double,std::vector<double>> out0(nnz, data4, index4, vector_info<double>());
	out0.allocate(a.non_zeros() + b.non_zeros() + c.non_zeros());
	a.template chip<size_t>(0, 0, outdim).fast_write(out0);
	b.template chip<size_t>(2, 0, outdim).fast_write(out0);
	c.template chip<size_t>(4, 0, outdim).fast_write_finalize(out0);
	std::vector<double> expect_data4 = {
		6, 9, 8, 7,
		3, 1,
		5, 7, 1,
		1, 7,
		2, 3,
		3
	};
	std::vector<size_t> expect_index4 = {
		0, 2, 3, 4,
		12, 13,
		14, 17, 19,
		22, 23,
		31, 32,
		36
	};
	ASSERT_EQ(nnz, expect_index4.size());
	EXPECT_ARREQ(expect_data4, data4);
	EXPECT_ARREQ(expect_index4, index4);
}


TEST(OPS, ChipWithZerosToSrc)
{
	// 9 8
	// 0 0
	// 0 7
	//
	// 7 0
	// 0 2
	// 0 0
	std::vector<double> data0 = { 9, 8, 7, 7, 2 };
	std::vector<size_t> indices0 = { 0, 1, 5, 6, 9 };

	SparseTensorSrc<double,3> a({2, 3, 2}, indices0.size(), indices0.data(), data0.data());

	// 6 0
	// 0 0
	// 5 0
	//
	// 0 1
	// 0 0
	// 0 3
	std::vector<double> data1 = { 6, 0, 0, 0, 5, 0, 0, 1, 0, 0, 0, 3 };
	std::vector<size_t> indices1(12);
	std::iota(indices1.begin(), indices1.end(), 0);

	SparseTensorSrc<double,3> b({2, 3, 2}, indices1.size(), indices1.data(), data1.data());

	// 7 0 0
	// 0 3 1
	// 0 1 0
	//
	// 0 0 0
	// 3 0 0
	// 0 0 0
	std::vector<double> data2 = { 7, 3, 1, 1, 3 };
	std::vector<size_t> indices2 = { 0, 4, 5, 7, 12 };

	SparseTensorSrc<double,3> c({3, 3, 2}, indices2.size(), indices2.data(), data2.data());

	// 9 8 6 0 7 0 0
	// 0 0 0 0 0 3 1
	// 0 7 5 0 0 1 0
	//
	// 7 0 0 1 0 0 0
	// 0 2 0 0 3 0 0
	// 0 0 0 3 0 0 0
	size_t nnz = 0;
	std::vector<double> data4;
	std::vector<size_t> index4;
	ShapeT<3> outdim = {7, 3, 2};
	SparseTensorDst<double,std::vector<double>> out0(nnz, data4, index4, vector_info<double>());
	out0.allocate(a.non_zeros() + b.non_zeros() + c.non_zeros());
	a.template chip<size_t>(0, 0, outdim).fast_write(out0);
	b.template chip<size_t>(2, 0, outdim).fast_write(out0);
	c.template chip<size_t>(4, 0, outdim).fast_write_finalize(out0);
	std::vector<double> expect_data4 = {
		9, 8, 6, 7,
		3, 1,
		7, 5, 1,
		7, 1,
		2, 3,
		3
	};
	std::vector<size_t> expect_index4 = {
		0, 1, 2, 4,
		12, 13,
		15, 16, 19,
		21, 24,
		29, 32,
		38
	};
	ASSERT_EQ(nnz, expect_index4.size());
	EXPECT_ARREQ(expect_data4, data4);
	EXPECT_ARREQ(expect_index4, index4);
}


TEST(OPS, ChipToTensor)
{
	// 6 0
	// 0 0
	// 5 0
	//
	// 0 1
	// 0 0
	// 0 3
	std::vector<double> data0 = { 6, 5, 1, 3 };
	std::vector<size_t> indices0 = { 0, 4, 7, 11 };

	SparseTensorSrc<double,3> a({2, 3, 2}, indices0.size(), indices0.data(), data0.data());

	// 9 8
	// 0 0
	// 0 7
	//
	// 7 0
	// 0 2
	// 0 0
	std::vector<double> data1 = { 9, 8, 7, 7, 2 };
	std::vector<size_t> indices1 = { 0, 1, 5, 6, 9 };

	SparseTensorSrc<double,3> b({2, 3, 2}, indices1.size(), indices1.data(), data1.data());

	// 7 0 0
	// 0 3 1
	// 0 1 0
	//
	// 0 0 0
	// 3 0 0
	// 0 0 0
	std::vector<double> data2 = { 7, 3, 1, 1, 3 };
	std::vector<size_t> indices2 = { 0, 4, 5, 7, 12 };

	SparseTensorSrc<double,3> c({3, 3, 2}, indices2.size(), indices2.data(), data2.data());

	// 6 3 9 8 7 3 3
	// 3 3 3 3 3 3 1
	// 5 3 3 7 3 1 3
	//
	// 3 1 7 3 3 3 3
	// 3 3 3 2 3 3 3
	// 3 3 3 3 3 3 3
	std::vector<double> dst_data(42);
	TensorMap<Tensor<double,3>> dst(dst_data.data(), {7, 3, 2});
	dst.setConstant(3);

	ShapeT<3> outdim = {7, 3, 2};
	a.template chip<size_t>(0, 0, outdim).write(dst);
	b.template chip<size_t>(2, 0, outdim).write(dst);
	c.template chip<size_t>(4, 0, outdim).write(dst);

	std::stringstream ss;
	ss << dst;
	EXPECT_STREQ("6 3 5 3 3 3\n3 3 3 1 3 3\n9 3 3 7 3 3\n8 3 7 3 2 3\n7 3 3 3 3 3\n3 3 1 3 3 3\n3 1 3 3 3 3",
		ss.str().c_str());
}
