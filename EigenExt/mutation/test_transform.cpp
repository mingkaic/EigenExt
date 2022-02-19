
#include "EigenExt/mutation/common.hpp"


template <typename T, size_t RANK, size_t RED_SIZE>
using TensReduceOpF = std::function<Tensor<T,RANK-RED_SIZE>(
	const Tensor<T,RANK>&,const std::array<int,RED_SIZE>&)>;


template <typename T, size_t RANK>
struct TransformMutator : public TensorMutator<T,RANK>
{
	virtual ~TransformMutator (void) = default;

	template <typename REDUCE_OP, size_t RED_SIZE>
	void generate_reduce_testcase (TensReduceOpF<T,RANK,RED_SIZE> op,
		T eps = std::numeric_limits<T>::epsilon())
	{
		std::vector<T> adata;
		std::vector<size_t> aindices;
		auto atens = this->get_sparse_tensor(adata, aindices, "a", this->get_shape("shape"), 0.5);

		auto rperm = this->permute_indices("reduce_indices", RANK, RED_SIZE);
		std::array<int,RED_SIZE> rindices;
		std::copy(rperm.begin(), rperm.end(), rindices.begin());

		Tensor<T,RANK-RED_SIZE> expect = op(atens, rindices);

		// register expected
		auto expect_ptr = expect.data();
		std::vector<T> expected_data(expect_ptr,
			expect_ptr + internal::array_prod(expect.dimensions()));
		this->write_entry("expected_data", expected_data);

		auto a = this->make_sparse_src(atens, adata, aindices);

		std::set<int> ranks(rindices.begin(), rindices.end());

		size_t nnz = 0;
		std::vector<T> outdata;
		std::vector<size_t> outindex;
		SparseTensorDst<T,std::vector<T>> out(nnz, outdata, outindex, vector_info<T>());
		a.template reduce<int, REDUCE_OP>(ranks).write(out);

		std::vector<T> got_data(expected_data.size(), 0);
		for (size_t i = 0; i < nnz; ++i)
		{
			got_data[outindex[i]] = outdata[i];
		}
		EXPECT_ARRCLOSE(expected_data, got_data, eps);
	}
};


struct TRANSFORM_2D_DOUB : public TransformMutator<double,2> {};


struct TRANSFORM_3D_DOUB : public TransformMutator<double,3> {};


static const double eps = std::numeric_limits<float>::epsilon();


template <typename T, size_t RANK, size_t RED_SIZE>
static Tensor<T,RANK-RED_SIZE> reduce_sum (
	const Tensor<T,RANK>& arg, const std::array<int,RED_SIZE>& dims)
{
	return arg.sum(dims);
}


template <typename T, size_t RANK, size_t RED_SIZE>
static Tensor<T,RANK-RED_SIZE> reduce_prod (
	const Tensor<T,RANK>& arg, const std::array<int,RED_SIZE>& dims)
{
	return arg.prod(dims);
}


template <typename T, size_t RANK, size_t RED_SIZE>
static Tensor<T,RANK-RED_SIZE> reduce_min (
	const Tensor<T,RANK>& arg, const std::array<int,RED_SIZE>& dims)
{
	return arg.minimum(dims);
}


template <typename T, size_t RANK, size_t RED_SIZE>
static Tensor<T,RANK-RED_SIZE> reduce_max (
	const Tensor<T,RANK>& arg, const std::array<int,RED_SIZE>& dims)
{
	return arg.maximum(dims);
}


TEST_F(TRANSFORM_2D_DOUB, ReduceSum_1Dims)
{
	generate_reduce_testcase<Eigen::SumReducer<double>,1>(reduce_sum<double,2,1>, eps);
}


TEST_F(TRANSFORM_2D_DOUB, ReduceSum_2Dims)
{
	generate_reduce_testcase<Eigen::SumReducer<double>,2>(reduce_sum<double,2,2>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceSum_1Dims)
{
	generate_reduce_testcase<Eigen::SumReducer<double>,1>(reduce_sum<double,3,1>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceSum_2Dims)
{
	generate_reduce_testcase<Eigen::SumReducer<double>,2>(reduce_sum<double,3,2>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceSum_3Dims)
{
	generate_reduce_testcase<Eigen::SumReducer<double>,3>(reduce_sum<double,3,3>, eps);
}


TEST_F(TRANSFORM_2D_DOUB, ReduceProd_1Dims)
{
	generate_reduce_testcase<Eigen::ProdReducer<double>,1>(reduce_prod<double,2,1>, eps);
}


TEST_F(TRANSFORM_2D_DOUB, ReduceProd_2Dims)
{
	generate_reduce_testcase<Eigen::ProdReducer<double>,2>(reduce_prod<double,2,2>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceProd_1Dims)
{
	generate_reduce_testcase<Eigen::ProdReducer<double>,1>(reduce_prod<double,3,1>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceProd_2Dims)
{
	generate_reduce_testcase<Eigen::ProdReducer<double>,2>(reduce_prod<double,3,2>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceProd_3Dims)
{
	generate_reduce_testcase<Eigen::ProdReducer<double>,3>(reduce_prod<double,3,3>, eps);
}


TEST_F(TRANSFORM_2D_DOUB, ReduceMin_1Dims)
{
	generate_reduce_testcase<Eigen::MinReducer<double>,1>(reduce_min<double,2,1>, eps);
}


TEST_F(TRANSFORM_2D_DOUB, ReduceMin_2Dims)
{
	generate_reduce_testcase<Eigen::MinReducer<double>,2>(reduce_min<double,2,2>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceMin_1Dims)
{
	generate_reduce_testcase<Eigen::MinReducer<double>,1>(reduce_min<double,3,1>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceMin_2Dims)
{
	generate_reduce_testcase<Eigen::MinReducer<double>,2>(reduce_min<double,3,2>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceMin_3Dims)
{
	generate_reduce_testcase<Eigen::MinReducer<double>,3>(reduce_min<double,3,3>, eps);
}


TEST_F(TRANSFORM_2D_DOUB, ReduceMax_1Dims)
{
	generate_reduce_testcase<Eigen::MaxReducer<double>,1>(reduce_max<double,2,1>, eps);
}


TEST_F(TRANSFORM_2D_DOUB, ReduceMax_2Dims)
{
	generate_reduce_testcase<Eigen::MaxReducer<double>,2>(reduce_max<double,2,2>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceMax_1Dims)
{
	generate_reduce_testcase<Eigen::MaxReducer<double>,1>(reduce_max<double,3,1>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceMax_2Dims)
{
	generate_reduce_testcase<Eigen::MaxReducer<double>,2>(reduce_max<double,3,2>, eps);
}


TEST_F(TRANSFORM_3D_DOUB, ReduceMax_3Dims)
{
	generate_reduce_testcase<Eigen::MaxReducer<double>,3>(reduce_max<double,3,3>, eps);
}
