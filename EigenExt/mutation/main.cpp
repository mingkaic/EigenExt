#include <chrono>

#include "gtest/gtest.h"

#include "muta/mutator.hpp"

int main (int argc, char** argv)
{
	muta::Mutator::generator_.seed(std::chrono::system_clock::now().time_since_epoch().count());

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
