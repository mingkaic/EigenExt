
#include "exam/exam.hpp"

#include "EigenExt/sparse_tensor.h"


using namespace Eigen;


TEST(OPS, CwiseBinarySparseSparseToSparse)
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
	a.cwiseBinary(b, [](const double& a, const double& b){ return a + b; }).write(out0);
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


TEST(OPS, CwiseBinarySparseSparseToDense_LeadRight)
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
	// 0 0 2 0
	std::vector<double> bdata = { 1, -9, 1, 2, 2, -4, 1, 2 };
	std::vector<size_t> bindices = { 1, 2, 6, 8, 12, 14, 19, 22 };

	SparseTensorSrc<double,3> b({4, 3, 2}, bindices.size(), bindices.data(), bdata.data());

	// 6 1 0 8
	// 0 0 1 0
	// 7 0 0 7
	//
	// 2 1 3 0
	// 0 0 0 3
	// 0 3 2 0
	std::vector<double> data0(24);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	a.cwiseBinary(b,[](const double& a, const double& b){ return a + b; }).write(out0);
	std::vector<double> expect_data = {
		6, 1, 0, 8,
		0, 0, 1, 0,
		7, 0, 0, 7,
		2, 1, 3, 0,
		0, 0, 0, 3,
		0, 3, 2, 0
	};
	EXPECT_ARREQ(expect_data, data0);
}


TEST(OPS, CwiseBinarySparseSparseToDense_LeadLeft)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 0 3 0
	std::vector<double> adata = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> aindices = { 0, 2, 3, 8, 11, 13, 14, 19, 22 };

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
	// 0 2 3 0
	std::vector<double> data0(24);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	a.cwiseBinary(b,[](const double& a, const double& b){ return a + b; }).write(out0);
	std::vector<double> expect_data = {
		6, 1, 0, 8,
		0, 0, 1, 0,
		7, 0, 0, 7,
		2, 1, 3, 0,
		0, 0, 0, 3,
		0, 2, 3, 0
	};
	EXPECT_ARREQ(expect_data, data0);
}


TEST(OPS, CwiseBinarySparseDenseToDense)
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

	// 15  -59	-75	-36
	// -1  72	76	61
	// 20  7	87	66
	//
	// -80 -82 -73	79
	// 8   -12 61	75
	// 8   73  70	41
	std::vector<double> bdata = {
		15, -59, -75, -36,
		-1, 72, 76, 61,
		20, 7, 87, 66,
		-80, -82, -73, 79,
		8, -12, 61, 75,
		8, 73, 70, 41
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> b(bdata.data(), {4, 3, 2});

	// 21 -59 -66 -28
	// -1 72 76 61
	// 25 7 87 73
	//
	// -80 -81 -67 79
	// 8 -12 61 77
	// 8 76 70 41
	std::vector<double> data0(24);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	a.cwiseBinary(b,[](const double& a, const double& b){ return a + b; }).write(out0);
	std::vector<double> expect_data = {
		21,-59,-66,-28,
		-1,72,76,61,
		25,7,87,73,

		-80,-81,-66,79,
		8,-12,61,77,
		8,76,70,41
	};
	EXPECT_ARREQ(expect_data, data0);
}


TEST(OPS, CwiseBinaryDenseSparseToDense)
{
	// 15  -59	-75	-36
	// -1  72	76	61
	// 20  7	87	66
	//
	// -80 -82 -73	79
	// 8   -12 61	75
	// 8   73  70	41
	std::vector<double> adata = {
		15, -59, -75, -36,
		-1, 72, 76, 61,
		20, 7, 87, 66,
		-80, -82, -73, 79,
		8, -12, 61, 75,
		8, 73, 70, 41
	};
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> a(adata.data(), {4, 3, 2});

	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<double> bdata = { 6, 9, 8, 5, 7, 1, 7, 2, 3 };
	std::vector<size_t> bindices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };

	SparseTensorSrc<double,3> b({4, 3, 2}, bindices.size(), bindices.data(), bdata.data());

	// 21 -59 -66 -28
	// -1 72 76 61
	// 25 7 87 73
	//
	// -80 -81 -67 79
	// 8 -12 61 77
	// 8 76 70 41
	std::vector<double> data0(24);
	::Eigen::TensorMap<::Eigen::Tensor<double,3>> out0(data0.data(), {4, 3, 2});
	b.cwiseRBinary(a,[](const double& a, const double& b){ return a + b; }).write(out0);
	std::vector<double> expect_data = {
		21,-59,-66,-28,
		-1,72,76,61,
		25,7,87,73,

		-80,-81,-66,79,
		8,-12,61,77,
		8,76,70,41
	};
	EXPECT_ARREQ(expect_data, data0);
}
