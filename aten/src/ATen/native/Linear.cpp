#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtilsMulti.h"


namespace at { namespace native {

Tensor sumproduct_pair(const Tensor& left_, const Tensor& right_, IntList sum_dims_, bool keepdim) {
  // assumes that tensors have been pre-unsqueezed
  AT_ASSERT(left_.dim()==right_.dim(), "number of dimensions must match");
  if (sum_dims_.size() == 0)
    return at::mul(left_, right_);
  int64_t dim = left_.dim();
  auto sum_dims = dim_list_to_vector(sum_dims_, dim);
  std::vector<int64_t> lro, lo, ro;
  int64_t lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
  Tensor left = left_;
  Tensor right = right_;
  for (int64_t i = 0; i < dim; i++) {
    auto sl = left.size(i)>1;
    auto sr = right.size(i)>1;
    if (sum_dims[i]) {
      if (sl && sr) {
	AT_ASSERT(left.size(i)==right.size(i), "sum indexes must match");
	sum_size *= left.size(i);
      } else if (sl) {
	left = left.sum(i, true);
      } else if (sr) {
	right = right.sum(i, true);
      }
    } else if (sl && sr) {
      AT_ASSERT(left.size(i)==right.size(i), "non-broadcast dimensions must match");
      lro.push_back(i);
      lro_size *= left.size(i);
    } else if (sl) {
      lo.push_back(i);
      lo_size *= left.size(i);
    } else {
      ro.push_back(i);
      ro_size *= right.size(i);
    }
  }
  std::vector<int64_t> out_size;
  for (auto& d : lro) out_size.push_back(left.size(d));
  for (auto& d : lo) out_size.push_back(left.size(d));
  for (auto& d : sum_dims_) { out_size.push_back(1); (void)(d); }; // avoid warining about not using d
  for (auto& d : ro) out_size.push_back(right.size(d));

  std::vector<int64_t> lpermutation(lro);
  lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
  lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

  std::vector<int64_t> rpermutation(lro);
  rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
  rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

  std::vector<int64_t> opermutation(lro.size()+lo.size()+sum_dims_.size()+ro.size(), -1);
  {
  int64_t i = 0;

  for (auto it = lro.begin(); it != lro.end(); i++, it++) {
    opermutation[*it] = i;
  }
  for (auto it = lo.begin(); it != lo.end(); i++, it++) {
    opermutation[*it] = i;
  }
  for (auto it = sum_dims_.begin(); it != sum_dims_.end(); i++, it++) {
    opermutation[*it] = i;
  }
  for (auto it = ro.begin(); it != ro.end(); i++, it++) {
    opermutation[*it] = i;
  }
  }

  left = left.permute(lpermutation).reshape({lro_size, lo_size, sum_size});
  right = right.permute(rpermutation).reshape({lro_size, sum_size, ro_size});
  Tensor result = at::bmm(left, right);
  result = result.view(out_size).permute(opermutation);
  if (! keepdim) {
    for (int i = dim-1; i>=0; i--)
      if (sum_dims[i])
	result.squeeze_(i);
  }
  return result;
}


Tensor einsum(std::string eqn, TensorList tensors) {
  std::string in_eqn;
  size_t pos;
  std::vector<std::int64_t> number_of_occurences(26, 0);
  std::vector<std::int64_t> last_occurence(26, -1);
  std::vector<std::int64_t> sorted_position(26, -1);
  if ((pos = eqn.find("->")) != std::string::npos) {
    in_eqn = eqn.substr(0, pos);
  } else {
    in_eqn = eqn;
  }

  int64_t operand = 0;
  std::stringstream eqn_stream(in_eqn);
  std::string term;
  while (! eqn_stream.eof()) {
    std::getline(eqn_stream, term, ',');
    int64_t dims_in_operand = 0;
    for (auto &c : term) {
      AT_ASSERT(('a' <= c) && (c <= 'z'), "only lowercase letters a-z allowed as indices");
      int64_t index_num = c-'a';
      number_of_occurences[index_num]++;
      AT_ASSERT(last_occurence[index_num] < operand, "diagonals (multiple occurences of the same index for one tensor) not implemented yet")
      last_occurence[index_num] = operand;
      dims_in_operand++;
    }
    AT_ASSERT((int64_t) tensors.size()>operand, "more operands in equation than tensors");
    AT_ASSERT(dims_in_operand == tensors[operand].dim(), "dimension mismatch for operand %zd: equation %zd, tensor %zd", operand, dims_in_operand, tensors[operand].dim());
    operand++;
  }
  AT_ASSERT((int64_t) tensors.size()==operand, "more tensors than operands in equation");

  int64_t num_outputs = 0;
  std::vector<int64_t> position_labels;
  if (pos != std::string::npos) {
    for (auto &c : eqn.substr(pos+2)) {
      AT_ASSERT(('a' <= c) && (c <= 'z'), "only lowercase letters a-z allowed as indices");
      int64_t index_num = c-'a';
      AT_ASSERT(sorted_position[index_num] == -1, "index %c occurs twice in output", c);
      sorted_position[index_num] = num_outputs;
      position_labels.push_back(index_num);
      num_outputs++;
    }
  } else {
    for (size_t idx = 0; idx < 26; idx++) {
      if (number_of_occurences[idx] == 1) {
	sorted_position[idx] = num_outputs;
	position_labels.push_back(idx);
	num_outputs++;
      }
    }
  }
  int64_t position = num_outputs;
  for (int64_t idx = 0; idx < 26; idx++) {
    if ((number_of_occurences[idx] > 0) && (sorted_position[idx]==-1)) {
      sorted_position[idx] = position;
      position_labels.push_back(idx);
      position++;
    }
  }
  std::vector<Tensor> permuted_ops;
  eqn_stream.clear();
  eqn_stream.seekg(0, std::ios_base::beg);
  for (int64_t op = 0; op < (int64_t) tensors.size(); op++) {
    std::vector<int64_t> axes(26, -1);
    std::vector<int64_t> permutation;
    std::getline(eqn_stream, term, ',');
    int64_t dim = 0;
    for (auto &c : term) {
      int64_t index_num = c-'a';
      axes[index_num] = dim;
      dim++;
    }
    for (auto &c : position_labels) {
      if (axes[c] > -1) {
	permutation.push_back(axes[c]);
      }
    }
    permuted_ops.push_back(tensors[op].permute(permutation));
    for (int64_t dim = 0; dim < (int64_t) position_labels.size(); dim++) {
      auto c = position_labels[dim];
      if (axes[c] == -1) {
	permuted_ops.back().unsqueeze_(dim);
      }
    }
  }
  // we just go left to right. numpy allows to optimize the path...
  Tensor result = permuted_ops[0];
  for (int64_t idx = 0; idx < 26; idx++) {
    if ((last_occurence[idx] == 0)
	&& (sorted_position[idx]>=num_outputs)) {
      result = result.sum(sorted_position[idx], true);
    }
  }

  for (int64_t i = 1; i < (int64_t) permuted_ops.size(); i++) {
    std::vector<int64_t> sum_dims;
    for (int64_t idx = 0; idx < 26; idx++) {
      if ((last_occurence[idx] == i)
	  && (sorted_position[idx]>=num_outputs)) {
	sum_dims.push_back(sorted_position[idx]);
      }
    }
    result = at::native::sumproduct_pair(result, permuted_ops[i], sum_dims, true);
  }
  for (int64_t dim = position_labels.size()-1; dim >= num_outputs; dim--)
    result.squeeze_(dim);
  return result;
}

// sumproduct_pair computes `(left*right).sum(sumdims)` by means of permutation and
// batch matrix multiplication // its main purpose is to provide a pairwise
// reduction for einsum
Tensor sumproduct_pair(const Tensor& left_, const Tensor& right_, IntList sum_dims_, bool keepdim) {
  // assumes that tensors have been pre-unsqueezed
  AT_ASSERT(left_.dim()==right_.dim(), "number of dimensions must match");
  if (sum_dims_.size() == 0)
    return at::mul(left_, right_);
  int64_t dim = left_.dim();
  auto sum_dims = dim_list_to_bitset(sum_dims_, dim);
  std::vector<int64_t> lro, lo, ro;
  int64_t lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
  Tensor left = left_;
  Tensor right = right_;
  for (int64_t i = 0; i < dim; i++) {
    auto sl = left.size(i)>1;
    auto sr = right.size(i)>1;
    if (sum_dims[i]) {
      if (sl && sr) {
	AT_ASSERT(left.size(i)==right.size(i), "sum indexes must match");
	sum_size *= left.size(i);
      } else if (sl) {
	left = left.sum(i, true);
      } else if (sr) {
	right = right.sum(i, true);
      }
    } else if (sl && sr) {
      AT_ASSERT(left.size(i)==right.size(i), "non-broadcast dimensions must match");
      lro.push_back(i);
      lro_size *= left.size(i);
    } else if (sl) {
      lo.push_back(i);
      lo_size *= left.size(i);
    } else {
      ro.push_back(i);
      ro_size *= right.size(i);
    }
  }
  std::vector<int64_t> out_size;
  for (auto& d : lro) out_size.push_back(left.size(d));
  for (auto& d : lo) out_size.push_back(left.size(d));
  for (auto& d : sum_dims_) { out_size.push_back(1); (void)(d); }; // avoid warining about not using d
  for (auto& d : ro) out_size.push_back(right.size(d));

  std::vector<int64_t> lpermutation(lro);
  lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
  lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

  std::vector<int64_t> rpermutation(lro);
  rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
  rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

  std::vector<int64_t> opermutation(lro.size()+lo.size()+sum_dims_.size()+ro.size(), -1);
  {
  int64_t i = 0;

  for (auto it = lro.begin(); it != lro.end(); i++, it++) {
    opermutation[*it] = i;
  }
  for (auto it = lo.begin(); it != lo.end(); i++, it++) {
    opermutation[*it] = i;
  }
  for (auto it = sum_dims_.begin(); it != sum_dims_.end(); i++, it++) {
    opermutation[*it] = i;
  }
  for (auto it = ro.begin(); it != ro.end(); i++, it++) {
    opermutation[*it] = i;
  }
  }

  left = left.permute(lpermutation).reshape({lro_size, lo_size, sum_size});
  right = right.permute(rpermutation).reshape({lro_size, sum_size, ro_size});
  Tensor result = at::bmm(left, right);
  result = result.view(out_size).permute(opermutation);
  if (! keepdim) {
    for (int i = dim-1; i>=0; i--)
      if (sum_dims[i])
	result.squeeze_(i);
  }
  return result;
}

// _trilinear computes a trilinear einstein sum with an unrolled dimension
// the result is `(i1.unsqueeze(expand1)*i2.unsqueeze(expand2)*i2.unsqueeze(expand3)).sum(sumdim)`
// the computation is unrolled in the unroll_dim dimension
// its main purpose is to unify the computations in bilinear and bilinear_backward
Tensor _trilinear(const Tensor& i1_, const Tensor& i2_, const Tensor& i3_,
		  IntList expand1_, IntList expand2_, IntList expand3_,
		  IntList sumdim_, int64_t unroll_dim) {
  int64_t total_dim = i1_.dim()+expand1_.size();
  AT_ASSERT((unroll_dim >= 0) && (unroll_dim < total_dim), "unroll_dim must be in [0,%zd]", total_dim-1);
  auto expand1 = dim_list_to_bitset(expand1_, total_dim);
  auto expand2 = dim_list_to_bitset(expand2_, total_dim);
  auto expand3 = dim_list_to_bitset(expand3_, total_dim);
  auto sumdim  = dim_list_to_bitset(sumdim_,  total_dim);
  Tensor i1 = i1_;
  Tensor i2 = i2_;
  Tensor i3 = i3_;
  std::vector<int64_t> output_size;
  std::vector<int64_t> sum_dims_12, sum_dims_23;
  int64_t unroll_size = -1;
  // asserts...
  for (int64_t i = 0; i < total_dim; i++) {
    int64_t s = 0;
    if (expand1[i]) {
      i1 = i1.unsqueeze(i);
    } else  {
      s = i1.size(i);
    }
    if (expand2[i]) {
      i2 = i2.unsqueeze(i);
    } else  {
      s = i2.size(i);
    }
    if (expand3[i]) {
      i3 = i3.unsqueeze(i);
      if (sumdim[i] && (i != unroll_dim))
	sum_dims_12.push_back(i);
    } else  {
      s = i3.size(i);
      if (sumdim[i] && (i != unroll_dim))
	sum_dims_23.push_back(i);
    }
    output_size.push_back(sumdim[i] ? 1 : s);
    if (i == unroll_dim)
      unroll_size = s;
  }
  int64_t slicemul1 = (expand1[unroll_dim] ? 0 : 1);
  int64_t slicemul2 = (expand2[unroll_dim] ? 0 : 1);
  int64_t slicemul3 = (expand3[unroll_dim] ? 0 : 1);

  auto output = i1.type().tensor(output_size).zero_();
  if (! sumdim[unroll_dim]) {
    for (int64_t k = 0; k < unroll_size; k++) {
      Tensor buf = at::native::sumproduct_pair(i1.narrow(unroll_dim, k * slicemul1, 1),
					       i2.narrow(unroll_dim, k * slicemul2, 1),
					       sum_dims_12, true);
      buf = at::native::sumproduct_pair(buf, i3.narrow(unroll_dim, k * slicemul3, 1), sum_dims_23, true);
      output.narrow(unroll_dim, k, 1).add_(buf);
    }
  }
  else {
    for (int64_t k = 0; k < unroll_size; k++) {
      Tensor buf = at::native::sumproduct_pair(i1.narrow(unroll_dim, k*slicemul1, 1),
					       i2.narrow(unroll_dim, k*slicemul2, 1), sum_dims_12, true);
      buf = at::native::sumproduct_pair(buf, i3.narrow(unroll_dim, k*slicemul3, 1), sum_dims_23, true);
      output.add_(buf);
    }
  }
  for (int64_t i = output.dim()-1; i >= 0; i--)
    if (sumdim[i])
      output.squeeze_(i);
  return output;
}

Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias) {
  AT_ASSERT(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got %lld and %lld",
            (long long)input1.dim(), (long long)input2.dim());
  for (int64_t i = 0; i < input1.dim() - 1; i++) {
    AT_ASSERT(input1.size(i) == input2.size(i),
              "bilinear(): input batch dimensions do not match at dim %lld: got %lld and %lld",
              (long long)i, (long long)input1.size(i), (long long)input2.size(i));
  }
  AT_ASSERT(input1.size(input1.dim() - 1) == weight.size(1),
            "bilinear(): input1 size does not match weight size: got %lld but expected %lld",
            (long long)input1.size(input1.dim() - 1), (long long)weight.size(1));
  AT_ASSERT(input2.size(input2.dim() - 1) == weight.size(2),
            "bilinear(): input2 size does not match weight size: got %lld but expected %lld",
            (long long)input2.size(input2.dim() - 1), (long long)weight.size(2));
  AT_ASSERT(!bias.defined() || bias.size(0) == weight.size(0),
            "bilinear(): bias size does not match weight size: got %lld but expected %lld",
            (long long)bias.size(0), (long long)weight.size(0));

  std::vector<int64_t> output_size;
  auto size1 = input1.sizes();
  output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
  output_size.push_back(weight.size(0));
  auto input1_flattened = input1.view({-1, input1.size(-1)});
  auto input2_flattened = input2.view({-1, input2.size(-1)});
  Tensor output = at::_trilinear(input1_flattened, weight, input2_flattened, {1,3}, {0}, {1,2}, {2,3}).reshape(output_size);
  if (bias.defined()) {
    output = output + bias;
  }
  return output;
}

}}  // namespace at::native
