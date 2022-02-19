
#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, AssignIndexToBuffer)
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

	// 0 0 0 8
	// 0 0 0 0
	// 0 0 0 7
	//
	// 0 1 0 0
	// 0 0 0 2
	// 0 3 0 0
	size_t nnz = 0;
	std::vector<double> data0;
	std::vector<size_t> index0;
	SparseTensorDst<double,std::vector<double>> out0(nnz, data0, index0, vector_info<double>());

	AccumulateBuffer<double,size_t> buffer;
	a.assignIndex(
	[](double& dst, const double& val, size_t idx)
	{
		if (idx % 2)
		{
			dst = val;
		}
	}).write(buffer);
	buffer.write(out0);

	std::vector<double> expect_data0 = {
		8, 7, 1, 2, 3,
	};
	std::vector<size_t> expect_index0 = {
		3, 11, 13, 19, 21
	};
	ASSERT_EQ(expect_index0.size(), nnz);
	EXPECT_ARREQ(expect_data0, data0);
	EXPECT_ARREQ(expect_index0, index0);
}


TEST(OPS, AssignIndexToTensor)
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

	// 0 0 0 8
	// 0 0 0 0
	// 0 0 0 7
	//
	// 0 1 0 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> dst_data(24);
	TensorMap<Tensor<double,3>> dst(dst_data.data(), {4, 3, 2});
	dst.setConstant(3);

	AccumulateBuffer<double,size_t> buffer;
	a.assignIndex(
	[](double& dst, const double& val, size_t idx)
	{
		if (idx % 2)
		{
			dst = val;
		}
	}).write(dst);

	std::stringstream ss;
	ss << dst;
	EXPECT_STREQ("3 3 3 3 3 3\n3 3 3 1 3 3\n3 3 3 3 3 3\n8 3 7 3 2 3", ss.str().c_str());
}
