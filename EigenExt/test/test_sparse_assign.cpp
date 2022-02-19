
#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, AssignToBuffer)
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

	// 0 1 -9 0
	// 0 0 1 0
	// 2 0 0 0
	//
	// 2 0 -4 0
	// 0 0 0 1
	// 0 2 0 0
	std::vector<double> bdata = { 1, -9, 1, 2, 2, -4, 1, 2 };
	std::vector<size_t> bindices = { 1, 2, 6, 8, 12, 14, 19, 21 };

	SparseTensorSrc<double,3> b({4, 3, 2}, bindices.size(), bindices.data(), bdata.data());

	// same as b
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());

	AccumulateBuffer<double,size_t> buffer;
	a.assign().write(buffer);
	b.assign().overwrite(buffer);
	buffer.write(out0);
	ASSERT_EQ(bdata.size(), nnz);
	EXPECT_ARREQ(bdata, data0);
	EXPECT_ARREQ(bindices, index0);
}


TEST(OPS, CwiseNnaryAddToBuffer)
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

	// 0 1 -9 0
	// 0 0 1 0
	// 2 0 0 0
	//
	// 2 0 -4 0
	// 0 0 0 1
	// 0 2 0 0
	std::vector<double> bdata = { 1, -9, 1, 2, 2, -4, 1, 2 };
	std::vector<size_t> bindices = { 1, 2, 6, 8, 12, 14, 19, 21 };

	SparseTensorSrc<double,3> b({4, 3, 2}, bindices.size(), bindices.data(), bdata.data());

	// 6 1 0 8
	// 0 0 1 0
	// 7 0 0 7
	//
	// 2 1 3 0
	// 0 0 0 3
	// 0 5 0 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());

	AccumulateBuffer<double,size_t> buffer;
	a.assign().write(buffer);
	b.assign([](const double& a, const double& b)
	{
		return a + b;
	}).write(buffer);
	buffer.write(out0);
	std::vector<double> expect_data0 = {
		6, 1, 8,
		1, 7, 7,
		2, 1, 3,
		3, 5
	};
	std::vector<size_t> expect_index0 = {
		0, 1, 3,
		6, 8, 11,
		12, 13, 14,
		19, 21
	};
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, CwiseNnaryMulToBuffer)
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

	// 0 1 2 0
	// 0 0 1 0
	// 2 0 0 0
	//
	// 2 0 2 0
	// 0 0 0 1
	// 0 2 0 0
	std::vector<double> bdata = { 1, 2, 1, 2, 2, 2, 1, 2 };
	std::vector<size_t> bindices = { 1, 2, 6, 8, 12, 14, 19, 21 };

	SparseTensorSrc<double,3> b({4, 3, 2}, bindices.size(), bindices.data(), bdata.data());

	// 0 0 18 0
	// 0 0 0 0
	// 10 0 0 0
	//
	// 0 0 14 0
	// 0 0 0 2
	// 0 6 0 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());

	AccumulateBuffer<double,size_t> buffer;
	a.assign().write(buffer);
	b.assign([](const double& a, const double& b)
	{
		return a * b;
	}).overwrite(buffer);
	buffer.write(out0);
	std::vector<double> expect_data0 = {
		18, 10,
		14, 2, 6
	};
	std::vector<size_t> expect_index0 = {
		2, 8,
		14, 19, 21,
	};
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, AssignToTensor)
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

	// same as a
	std::vector<double> dst_data(24);
	TensorMap<Tensor<double,3>> dst(dst_data.data(), {4, 3, 2});
	dst.setConstant(3);

	a.assign().overwrite(dst);

	std::stringstream ss;
	ss << dst;
	EXPECT_STREQ("6 0 5 0 0 0\n0 0 0 1 0 3\n9 0 0 7 0 0\n8 0 7 0 2 0", ss.str().c_str());
}


TEST(OPS, CwiseNnaryAddToTensor)
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

	// 0 1 -9 0
	// 0 0 1 0
	// 2 0 0 0
	//
	// 2 0 -4 0
	// 0 0 0 1
	// 0 2 0 0
	std::vector<double> bdata = { 1, -9, 1, 2, 2, -4, 1, 2 };
	std::vector<size_t> bindices = { 1, 2, 6, 8, 12, 14, 19, 21 };

	SparseTensorSrc<double,3> b({4, 3, 2}, bindices.size(), bindices.data(), bdata.data());

	// 6 1 0 8
	// 0 0 1 0
	// 7 0 0 7
	//
	// 2 1 3 0
	// 0 0 0 3
	// 0 5 0 0
	std::vector<double> dst_data(24);
	TensorMap<Tensor<double,3>> dst(dst_data.data(), {4, 3, 2});
	dst.setConstant(0);

	a.assign().write(dst);
	b.assign([](const double& a, const double& b)
	{
		return a + b;
	}).write(dst);

	std::stringstream ss;
	ss << dst;
	EXPECT_STREQ("6 0 7 2 0 0\n1 0 0 1 0 5\n0 1 0 3 0 0\n8 0 7 0 3 0", ss.str().c_str());
}


TEST(OPS, CwiseNnaryMulToTensor)
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

	// 0 1 2 0
	// 0 0 1 0
	// 2 0 0 0
	//
	// 2 0 2 0
	// 0 0 0 1
	// 0 2 0 0
	std::vector<double> bdata = { 1, 2, 1, 2, 2, 2, 1, 2 };
	std::vector<size_t> bindices = { 1, 2, 6, 8, 12, 14, 19, 21 };

	SparseTensorSrc<double,3> b({4, 3, 2}, bindices.size(), bindices.data(), bdata.data());

	// 0 0 18 0
	// 0 0 0 0
	// 10 0 0 0
	//
	// 0 0 14 0
	// 0 0 0 2
	// 0 6 0 0
	std::vector<double> dst_data(24);
	TensorMap<Tensor<double,3>> dst(dst_data.data(), {4, 3, 2});
	dst.setConstant(0);

	a.assign().write(dst);
	b.assign([](const double& a, const double& b)
	{
		return a * b;
	}).overwrite(dst);

	std::stringstream ss;
	ss << dst;
	EXPECT_STREQ(" 0  0 10  0  0  0\n 0  0  0  0  0  6\n18  0  0 14  0  0\n 0  0  0  0  2  0",
		ss.str().c_str());
}
