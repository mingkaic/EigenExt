#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(SRC, Creation)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	std::vector<double> data = { 6, 9, 8, 5, 7 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11 };

	SparseTensorSrc<double,2> a({4, 3}, indices.size(), indices.data(), data.data());

	auto aindices = a.get_indices();
	EXPECT_ARREQ(indices, aindices);

	auto aptr = a.data();
	std::vector<double> adata(aptr, aptr + a.non_zeros());
	EXPECT_ARREQ(data, adata);
}


TEST(SRC, DstCreation)
{
	// 6 0 9
	// 8 0 0
	// 0 0 5
	// 0 0 7
	size_t nnz = 5;
	std::vector<double> data = { 6, 9, 8, 5, 7 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11 };

	SparseTensorDst<double,std::vector<double>> dst(nnz, data, indices, vector_info<double>());

	SparseTensorSrc<double,2> a({3, 4}, dst);

	auto aindices = a.get_indices();
	EXPECT_ARREQ(indices, aindices);

	auto aptr = a.data();
	std::vector<double> adata(aptr, aptr + a.non_zeros());
	EXPECT_ARREQ(data, adata);
}


TEST(SRC, WriteToDst)
{
	size_t nnz = 0;
	std::vector<double> dst_data;
	std::vector<size_t> dst_indices;
	SparseTensorDst<double,std::vector<double>> dst(nnz, dst_data, dst_indices, vector_info<double>());

	// 6 0 9
	// 8 0 0
	// 0 0 5
	// 0 0 7
	std::vector<double> data = { 6, 9, 8, 5, 7 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11 };
	SparseTensorSrc<double,2> a({3, 4},
		indices.size(), indices.data(), data.data());

	a.write(dst);

	EXPECT_EQ(dst_data, data);
	EXPECT_EQ(dst_indices, indices);
	EXPECT_EQ(indices.size(), nnz);
}


TEST(SRC, WriteToTensor)
{
	std::vector<double> dst_data(12);
	TensorMap<Tensor<double,2>> dst(dst_data.data(), {3, 4});
	dst.setConstant(3);

	// 6 0 9
	// 8 0 0
	// 0 0 5
	// 0 0 7
	std::vector<double> data = { 6, 9, 8, 5, 7 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11 };
	SparseTensorSrc<double,2> a({3, 4},
		indices.size(), indices.data(), data.data());

	a.write(dst);

	std::stringstream ss;
	ss << dst;
	EXPECT_STREQ("6 8 3 3\n3 3 3 3\n9 3 5 7", ss.str().c_str());
}


TEST(SRC, OverwriteToTensor)
{
	std::vector<double> dst_data(12);
	TensorMap<Tensor<double,2>> dst(dst_data.data(), {3, 4});
	dst.setConstant(3);

	// 6 0 9
	// 8 0 0
	// 0 0 5
	// 0 0 7
	std::vector<double> data = { 6, 9, 8, 5, 7 };
	std::vector<size_t> indices = { 0, 2, 3, 8, 11 };
	SparseTensorSrc<double,2> a({3, 4},
		indices.size(), indices.data(), data.data());

	a.overwrite(dst);

	std::stringstream ss;
	ss << dst;
	EXPECT_STREQ("6 8 0 0\n0 0 0 0\n9 0 5 7", ss.str().c_str());
}
