#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(DST, Allocation)
{
	size_t nnzs = 0;
	std::vector<double> data;
	std::vector<size_t> indices;
	SparseTensorDst<double,std::vector<double>> a(nnzs, data, indices, vector_info<double>());

	a.allocate(5);

	EXPECT_EQ(5, indices.size());
	EXPECT_EQ(5, data.size());
	EXPECT_EQ(0, nnzs);
	EXPECT_EQ(data.data(), a.get_data());
	EXPECT_EQ(indices.data(), a.get_indices());
	EXPECT_EQ(indices.size(), a.alloc_size());
}


TEST(DST, ReAllocation)
{
	size_t nnzs = 0;
	std::vector<double> data;
	std::vector<size_t> indices;
	SparseTensorDst<double,std::vector<double>> a(nnzs, data, indices, vector_info<double>());

	a.allocate(5);
	a.allocate(3);

	EXPECT_EQ(5, indices.size());
	EXPECT_EQ(5, data.size());
	EXPECT_EQ(indices.size(), a.alloc_size());

	a.allocate(5);

	EXPECT_EQ(5, indices.size());
	EXPECT_EQ(5, data.size());
	EXPECT_EQ(indices.size(), a.alloc_size());

	a.allocate(6);

	EXPECT_EQ(6, indices.size());
	EXPECT_EQ(6, data.size());
	EXPECT_EQ(indices.size(), a.alloc_size());
}


TEST(DST, SetNnz)
{
	size_t nnzs = 0;
	std::vector<double> data;
	std::vector<size_t> indices;
	SparseTensorDst<double,std::vector<double>> a(nnzs, data, indices, vector_info<double>());

	a.set_nnz(5);

	EXPECT_EQ(5, indices.size());
	EXPECT_EQ(5, data.size());
	EXPECT_EQ(5, nnzs);
	EXPECT_EQ(5, a.non_zeros());
}


TEST(DST, OverAllocateOptimize)
{
	size_t nnzs = 0;
	std::vector<double> data;
	std::vector<size_t> indices;
	SparseTensorDst<double,std::vector<double>> a(nnzs, data, indices, vector_info<double>());

	a.allocate(10);
	a.set_nnz(5);

	EXPECT_EQ(10, indices.size());
	EXPECT_EQ(10, data.size());
	EXPECT_EQ(5, nnzs);
	EXPECT_EQ(5, a.non_zeros());
}
