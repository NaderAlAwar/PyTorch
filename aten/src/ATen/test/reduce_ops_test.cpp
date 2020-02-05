#include <gtest/gtest.h>

#include <torch/types.h>
#include <torch/utils.h>

using namespace at;

TEST(ReduceOpsTest, MaxValuesAndMinValues) {
  const int W = 10;
  const int H = 10;
  if (hasCUDA()) {
    for (const auto dtype : {kHalf, kFloat, kDouble, kShort, kInt, kLong}) {
      auto a = at::rand({H, W}, TensorOptions(kCUDA).dtype(at::kHalf));
      ASSERT_FLOAT_EQ(
        a.max_values(c10::IntArrayRef{0, 1}).item<double>(),
        a.max().item<double>()
      );
      ASSERT_FLOAT_EQ(
        a.min_values(c10::IntArrayRef{0, 1}).item<double>(),
        a.min().item<double>()
      );
      ASSERT_FLOAT_EQ(
        a.neg().max_values(c10::IntArrayRef{0, 1}).item<double>(),
        a.neg().max().item<double>()
      );
      ASSERT_FLOAT_EQ(
        a.neg().min_values(c10::IntArrayRef{0, 1}).item<double>(),
        a.neg().min().item<double>()
      );
    }
  }
}
