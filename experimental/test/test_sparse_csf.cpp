
#include <algorithm>
#include <random>

#include "exam/exam.hpp"

#include "EigenExt/sparse_csf.h"


using namespace Eigen;


template <size_t RANK>
static ShapeT<RANK> random_indices (
	std::vector<size_t>& indices, size_t maxdim)
{
	std::random_device rnd_device;
	std::mt19937 mersenne_engine(rnd_device());
	ShapeT<RANK> shape;

	std::uniform_int_distribution<size_t> dist(1, maxdim);
	std::generate(shape.begin(), shape.end(),
	[&dist, &mersenne_engine]()
	{
		return dist(mersenne_engine);
	});

	size_t n = internal::array_prod(shape);
	std::vector<size_t> data(n);

	std::uniform_int_distribution<size_t> bindist(0, 1);
	std::generate(data.begin(), data.end(),
	[&bindist, &mersenne_engine]()
	{
		return bindist(mersenne_engine);
	});

	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		if (data[i] == 1)
		{
			indices.push_back(i);
		}
	}
	return shape;
}


TEST(CSF, Expected2D)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	std::vector<size_t> indices = { 0, 2, 3, 8, 11 };
	csf::ShapeInfo<2> sinfo(ShapeT<2>{4, 3}, indices.data(), indices.size());
	// expected [
	//	{nzcount: [0, 3, 5], nzs: [0, 2, 3, 0, 3]},
	//	{nzcount: [0, 2], nzs: [0, 2]}
	// ]

	std::vector<size_t> expect_nzc0 = {0, 3, 5};
	std::vector<size_t> expect_nzs0 = {0, 2, 3, 0, 3};
	std::vector<size_t> expect_nzc1 = {0, 2};
	std::vector<size_t> expect_nzs1 = {0, 2};
	EXPECT_ARREQ(expect_nzc0, sinfo[0].nz_counts_);
	EXPECT_ARREQ(expect_nzs0, sinfo[0].nz_indices_);
	EXPECT_ARREQ(expect_nzc1, sinfo[1].nz_counts_);
	EXPECT_ARREQ(expect_nzs1, sinfo[1].nz_indices_);
}


TEST(CSF, Expected3D)
{
	// 6 0 9 8
	// 0 0 0 0
	// 5 0 0 7
	//
	// 0 1 7 0
	// 0 0 0 2
	// 0 3 0 0
	std::vector<size_t> indices = { 0, 2, 3, 8, 11, 13, 14, 19, 21 };
	csf::ShapeInfo<3> sinfo(ShapeT<3>{4, 3, 2}, indices.data(), indices.size());
	// expected [
	//	{nzcount: [0, 3, 5, 7, 8, 9], nzs: [0, 2, 3, 0, 3, 1, 2, 3, 1]},
	//	{nzcount: [0, 2, 5], nzs: [0, 2, 0, 1, 2]},
	//	{nzcount: [0, 2], nzs: [0, 1]}
	// ]

	std::vector<size_t> expect_nzc0 = {0, 3, 5, 7, 8, 9};
	std::vector<size_t> expect_nzs0 = {0, 2, 3, 0, 3, 1, 2, 3, 1};
	std::vector<size_t> expect_nzc1 = {0, 2, 5};
	std::vector<size_t> expect_nzs1 = {0, 2, 0, 1, 2};
	std::vector<size_t> expect_nzc2 = {0, 2};
	std::vector<size_t> expect_nzs2 = {0, 1};
	EXPECT_ARREQ(expect_nzc0, sinfo[0].nz_counts_);
	EXPECT_ARREQ(expect_nzs0, sinfo[0].nz_indices_);
	EXPECT_ARREQ(expect_nzc1, sinfo[1].nz_counts_);
	EXPECT_ARREQ(expect_nzs1, sinfo[1].nz_indices_);
	EXPECT_ARREQ(expect_nzc2, sinfo[2].nz_counts_);
	EXPECT_ARREQ(expect_nzs2, sinfo[2].nz_indices_);
}


TEST(CSF, Random2D)
{
	std::vector<size_t> expected_indices;
	auto shape = random_indices<2>(expected_indices, 144);
	csf::ShapeInfo<2> sinfo(shape, expected_indices.data(), expected_indices.size());

	std::vector<size_t> indices = sinfo.decode(shape);
	EXPECT_ARREQ(expected_indices, indices);
}


TEST(CSF, Random3D)
{
	std::vector<size_t> expected_indices;
	auto shape = random_indices<3>(expected_indices, 24);
	csf::ShapeInfo<3> sinfo(shape, expected_indices.data(), expected_indices.size());

	std::vector<size_t> indices = sinfo.decode(shape);
	EXPECT_ARREQ(expected_indices, indices);
}


TEST(CSF, Random4D)
{
	std::vector<size_t> expected_indices;
	auto shape = random_indices<4>(expected_indices, 12);
	csf::ShapeInfo<4> sinfo(shape, expected_indices.data(), expected_indices.size());

	std::vector<size_t> indices = sinfo.decode(shape);
	EXPECT_ARREQ(expected_indices, indices);
}
