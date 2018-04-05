/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/operators/unique_ops.h"

#include <cmath>
#include <google/dense_hash_map>

namespace caffe2 {

template <>
template <typename T>
bool UniqueOp<CPUContext>::DoRunWithType() {
  auto& inputTensor = Input(0);
  // use dim32 to enforce that it's fine to have remapping of type int
  int N = inputTensor.dim32(0);
  CAFFE_ENFORCE_EQ(inputTensor.ndim(), 1, "Input should be a vector");
  auto* uniqueTensor = Output(UNIQUE);

  int* remapping = nullptr;
  if (REMAPPING < OutputSize()) {
    auto* remappingTensor = Output(REMAPPING);
    remappingTensor->ResizeLike(inputTensor);
    remapping = remappingTensor->template mutable_data<int>();
  }

  const T* input = inputTensor.template data<T>();
  // TODO(dzhulgakov): if perf becomes an issue consider doing hash table
  // instead of sorting
  order_.resize(N);
  std::iota(order_.begin(), order_.end(), 0);
  std::sort(order_.begin(), order_.end(), [input](const int x, const int y) {
    return input[x] < input[y];
  });
  int K = N;
  for (int i = 1; i < N; ++i) {
    K -= input[order_[i]] == input[order_[i - 1]];
  }
  uniqueTensor->Resize(K);
  T* unique = uniqueTensor->template mutable_data<T>();
  K = 0;
  T prev = -1;
  for (int i = 0; i < N; ++i) {
    if (i == 0 || prev != input[order_[i]]) {
      prev = unique[K++] = input[order_[i]];
    }
    if (remapping) {
      remapping[order_[i]] = K - 1;
    }
  }
  return true;
}

template <>
template <typename T>
bool SparseHashUniqueOp<CPUContext>::DoRunWithType() {
  auto& inputTensor = Input(0);
  // use dim32 to enforce that it's fine to have remapping of type int
  int N = inputTensor.dim32(0);
  CAFFE_ENFORCE_EQ(inputTensor.ndim(), 1, "Input should be a vector");
  auto* uniqueTensor = Output(UNIQUE);

  int* remapping = nullptr;
  if (REMAPPING < OutputSize()) {
    auto* remappingTensor = Output(REMAPPING);
    remappingTensor->ResizeLike(inputTensor);
    remapping = remappingTensor->template mutable_data<int>();
  }

  auto* input = inputTensor.template data<T>();
  google::dense_hash_map<T, int> hashMap(N);

  // We assume input tensor only contains positive values
  hashMap.set_empty_key(-1);
  int K = 0;
  for (int i = 0; i < N; ++i) {
    CAFFE_ENFORCE_GE(
        input[i], 0, "SparseHashUnique assumes inputs are positive numbers");
    auto p = hashMap.insert(std::make_pair(input[i], K));
    if (p.second) {
      ++K;
    }
    if (remapping) {
      remapping[i] = p.first->second;
    }
  }
  uniqueTensor->Resize(K);
  auto* unique = uniqueTensor->template mutable_data<T>();
  for (auto p : hashMap) {
    unique[p.second] = p.first;
  }
  return true;
}

REGISTER_CPU_OPERATOR(Unique, UniqueOp<CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Unique,
    SparseHash,
    SparseHashUniqueOp<CPUContext>);

OPERATOR_SCHEMA(Unique)
    .NumInputs(1)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(
Deduplicates input indices vector and optionally produces reverse remapping.
There's no guarantees on the ordering of the output indices. When `SparseHash`
Engine is used, the input indices need to be non-negative.
)DOC")
    .Input(0, "indices", "1D tensor of int32 or int64 indices.")
    .Output(0, "unique_indices", "1D tensor of deduped entries.")
    .Output(
        1,
        "remapping",
        "(optional) mapping from `indices` to `unique_indices`. This has the "
        "same shape as `indices`. Its elements are the indices into "
        "`unique_indices` such that `Gather(['unique_indices', 'remapping'])` "
        "yields `indices`.")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());
      CAFFE_ENFORCE_EQ(in[0].dims_size(), 1);
      if (in[0].dims(0) <= 1) {
        // This special case is useful in some situation, e.g., when feeding
        // tensor inference with empty tensor (where the first dim is the batch
        // size)
        out[0].add_dims(in[0].dims(0));
      } else {
        out[0].set_unknown_shape(true);
      }
      if (def.output_size() > 1) {
        // Remapping has the same shape as the input tensor
        out.push_back(in[0]);
        out.back().set_data_type(TensorProto::INT32);
      }
      return out;
    });

SHOULD_NOT_DO_GRADIENT(Unique);

} // namespace caffe2
