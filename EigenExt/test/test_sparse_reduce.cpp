#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, ReduceSum)
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

	// 6 1 16 8
	// 0 0 0 2
	// 5 3 0 7
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.reduce<size_t,SumReducer<double>>({2}).write(out0);
	std::vector<double> expect_data0 = { 6, 1, 16, 8, 2, 5, 3, 7 };
	std::vector<size_t> expect_index0 = { 0, 1, 2, 3, 7, 8, 9, 11 };
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);

	// 11 0 9 15
	// 0 4 7 2
	nnz = 0;
	std::vector<double> data1;
	std::vector<size_t> index1;
	SparseTensorDst<double,std::vector<double>> out1(nnz, data1, index1, vector_info<double>());
	a.reduce<size_t,SumReducer<double>>({1}).write(out1);
	std::vector<double> expect_data1 = { 11, 9, 15, 4, 7, 2 };
	std::vector<size_t> expect_index1 = { 0, 2, 3, 5, 6, 7 };
	ASSERT_EQ(expect_index1.size(), nnz);
	EXPECT_ARREQ(expect_data1, data1);
	EXPECT_ARREQ(expect_index1, index1);

	// 23 0 12
	// 8 2 3
	nnz = 0;
	std::vector<double> data2;
	std::vector<size_t> index2;
	SparseTensorDst<double,std::vector<double>> out2(nnz, data2, index2, vector_info<double>());
	a.reduce<size_t,SumReducer<double>>({0}).write(out2);
	std::vector<double> expect_data2 = { 23, 12, 8, 2, 3 };
	std::vector<size_t> expect_index2 = { 0, 2, 3, 4, 5 };
	ASSERT_EQ(expect_index2.size(), nnz);
	EXPECT_ARREQ(expect_data2, data2);
	EXPECT_ARREQ(expect_index2, index2);
}


TEST(OPS, ReduceProd)
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

	// 0 0 63 0
	// 0 0 0 0
	// 0 0 0 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.reduce<size_t,ProdReducer<double>>({2}).write(out0);
	std::vector<double> expect_data0 = { 63 };
	std::vector<size_t> expect_index0 = { 2 };
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);

	// 0 0 0 0
	// 0 0 0 0
	nnz = 0;
	std::vector<double> data1;
	std::vector<size_t> index1;
	SparseTensorDst<double,std::vector<double>> out1(nnz, data1, index1, vector_info<double>());
	a.reduce<size_t,ProdReducer<double>>({1}).write(out1);
	std::vector<double> expect_data1 = {};
	std::vector<size_t> expect_index1 = {};
	ASSERT_EQ(expect_index1.size(), nnz);
	EXPECT_ARREQ(expect_data1, data1);
	EXPECT_ARREQ(expect_index1, index1);

	// 0 0 0
	// 0 0 0
	nnz = 0;
	std::vector<double> data2;
	std::vector<size_t> index2;
	SparseTensorDst<double,std::vector<double>> out2(nnz, data2, index2, vector_info<double>());
	a.reduce<size_t,ProdReducer<double>>({0}).write(out2);
	std::vector<double> expect_data2 = {};
	std::vector<size_t> expect_index2 = {};
	ASSERT_EQ(expect_index2.size(), nnz);
	EXPECT_ARREQ(expect_data2, data2);
	EXPECT_ARREQ(expect_index2, index2);
}


TEST(OPS, ReduceMin)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 -1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> data = { 6, 9, 8, 5, 7, -1, 7, 2, 3 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };

	SparseTensorSrc<double,3> a({4, 3, 2}, indices.size(), indices.data(), data.data());

	// 0 -1 7 0
	// 0 0 0 0
	// 0 0 0 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.reduce<size_t,MinReducer<double>>({2}).write(out0);
	std::vector<double> expect_data0 = { -1, 7 };
	std::vector<size_t> expect_index0 = { 1, 2 };
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);

	// 0 0 0 0
	// 0 -1 0 0
	nnz = 0;
	std::vector<double> data1;
	std::vector<size_t> index1;
	SparseTensorDst<double,std::vector<double>> out1(nnz, data1, index1, vector_info<double>());
	a.reduce<size_t,MinReducer<double>>({1}).write(out1);
	std::vector<double> expect_data1 = {-1};
	std::vector<size_t> expect_index1 = {5};
	ASSERT_EQ(expect_index1.size(), nnz);
	EXPECT_ARREQ(expect_data1, data1);
	EXPECT_ARREQ(expect_index1, index1);

	// 0 0 0
	// -1 0 0
	nnz = 0;
	std::vector<double> data2;
	std::vector<size_t> index2;
	SparseTensorDst<double,std::vector<double>> out2(nnz, data2, index2, vector_info<double>());
	a.reduce<size_t,MinReducer<double>>({0}).write(out2);
	std::vector<double> expect_data2 = {-1};
	std::vector<size_t> expect_index2 = {3};
	ASSERT_EQ(expect_index2.size(), nnz);
	EXPECT_ARREQ(expect_data2, data2);
	EXPECT_ARREQ(expect_index2, index2);
}


TEST(OPS, ReduceMax)
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

	// 6 1 9 8
	// 0 0 0 2
	// 5 3 0 7
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.reduce<size_t,MaxReducer<double>>({2}).write(out0);
	std::vector<double> expect_data0 = { 6, 1, 9, 8, 2, 5, 3, 7 };
	std::vector<size_t> expect_index0 = { 0, 1, 2, 3, 7, 8, 9, 11 };
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);

	// 6 0 9 8
	// 0 3 7 2
	nnz = 0;
	std::vector<double> data1;
	std::vector<size_t> index1;
	SparseTensorDst<double,std::vector<double>> out1(nnz, data1, index1, vector_info<double>());
	a.reduce<size_t,MaxReducer<double>>({1}).write(out1);
	std::vector<double> expect_data1 = {6, 9, 8, 3, 7, 2};
	std::vector<size_t> expect_index1 = {0, 2, 3, 5, 6, 7};
	ASSERT_EQ(expect_index1.size(), nnz);
	EXPECT_ARREQ(expect_data1, data1);
	EXPECT_ARREQ(expect_index1, index1);

	// 9 0 7
	// 7 2 3
	nnz = 0;
	std::vector<double> data2;
	std::vector<size_t> index2;
	SparseTensorDst<double,std::vector<double>> out2(nnz, data2, index2, vector_info<double>());
	a.reduce<size_t,MaxReducer<double>>({0}).write(out2);
	std::vector<double> expect_data2 = {9, 7, 7, 2, 3};
	std::vector<size_t> expect_index2 = {0, 2, 3, 4, 5};
	ASSERT_EQ(expect_index2.size(), nnz);
	EXPECT_ARREQ(expect_data2, data2);
	EXPECT_ARREQ(expect_index2, index2);
}


TEST(OPS, ArgMax)
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

	// 0 1 0 0
	// : : : 1
	// 0 1 : 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.reduce<size_t,ArgMaxReducer<double>>({2}).write(out0);
	std::vector<double> expect_data0 = { 1, 1, 1 };
	std::vector<size_t> expect_index0 = { 1, 7, 9 };
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);

	// 0 : 0 0
	// : 2 0 1
	nnz = 0;
	std::vector<double> data1;
	std::vector<size_t> index1;
	SparseTensorDst<double,std::vector<double>> out1(nnz, data1, index1, vector_info<double>());
	a.reduce<size_t,ArgMaxReducer<double>>({1}).write(out1);
	std::vector<double> expect_data1 = { 2, 1 };
	std::vector<size_t> expect_index1 = { 5, 7 };
	ASSERT_EQ(expect_index1.size(), nnz);
	EXPECT_ARREQ(expect_data1, data1);
	EXPECT_ARREQ(expect_index1, index1);

	// 2 : 3
	// 2 3 1
	nnz = 0;
	std::vector<double> data2;
	std::vector<size_t> index2;
	SparseTensorDst<double,std::vector<double>> out2(nnz, data2, index2, vector_info<double>());
	a.reduce<size_t,ArgMaxReducer<double>>({0}).write(out2);
	std::vector<double> expect_data2 = { 2, 3, 2, 3, 1 };
	std::vector<size_t> expect_index2 = { 0, 2, 3, 4, 5 };
	ASSERT_EQ(expect_index2.size(), nnz);
	EXPECT_ARREQ(expect_data2, data2);
	EXPECT_ARREQ(expect_index2, index2);
}


TEST(OPS, ReduceSumNoDimsIdentity)
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
	a.reduce<size_t,SumReducer<double>>({}).write(out0);
	ASSERT_EQ(indices.size(), nnz);
	EXPECT_ARREQ(data, data0);
	EXPECT_ARREQ(indices, index0);
}


TEST(OPS, ReduceSumOneDimsIdentity)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	std::vector<double> data = { 6, 9, 8, 5, 7 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11 };

	SparseTensorSrc<double,3> a({4, 3, 1}, indices.size(), indices.data(), data.data());

	// 6 1 16 8
	// 0 0 0 2
	// 5 3 0 7
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.reduce<size_t,SumReducer<double>>({2}).write(out0);
	ASSERT_EQ(indices.size(), nnz);
	EXPECT_ARREQ(data, data0);
	EXPECT_ARREQ(indices, index0);
}


TEST(OPS, ReduceSumLastDimsIdentity)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	std::vector<double> data = { 6, 9, 8, 5, 7 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11 };

	SparseTensorSrc<double,3> a({4, 3, 1}, indices.size(), indices.data(), data.data());

	// 6 1 16 8
	// 0 0 0 2
	// 5 3 0 7
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());
	a.reduce<size_t,SumReducer<double>>({2}).write(out0);
	ASSERT_EQ(indices.size(), nnz);
	EXPECT_ARREQ(data, data0);
	EXPECT_ARREQ(indices, index0);
}
