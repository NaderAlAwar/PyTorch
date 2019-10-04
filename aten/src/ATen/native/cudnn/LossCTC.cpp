#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>
#if AT_CUDNN_ENABLED()
  #include <ATen/cudnn/Descriptors.h>
#endif


#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

std::tuple<Tensor, Tensor> _cudnn_ctc_loss(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, bool deterministic, bool zero_infinity) {
  AT_ERROR("cudnn_ctc_loss: ATen not compiled with cuDNN >= 7 support");
}

}}

#else // AT_CUDNN_ENABLED

#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

#include <ATen/TensorUtils.h>

namespace at { namespace native {

namespace {

}  // namespace

std::tuple<Tensor, Tensor> _cudnn_ctc_loss(const Tensor& log_probs_t, const Tensor& targets_t, IntArrayRef input_lengths_, IntArrayRef target_lengths_, int64_t BLANK, bool deterministic, bool zero_infinity) {
  (void)zero_infinity; // only used for backward
  CheckedFrom c = "cudnn_ctc_loss";
  TensorArg log_probs { log_probs_t, "log_probs", 1 };
  TensorArg targets { targets_t, "targets", 2 };
  checkDim(c, log_probs, 3);
  checkScalarType(c, log_probs, kFloat);
  checkDim(c, targets, 1);
  checkScalarType(c, targets, kInt);
  checkContiguous(c, targets); // ?
  checkBackend(c, {*log_probs}, Backend::CUDA);
  checkBackend(c, {*targets}, Backend::CPU);
  int64_t batch_size = log_probs->size(1);
  TORCH_CHECK(input_lengths_.size() == batch_size, "input_lengths needs to have size to match batch_size");
  TORCH_CHECK(target_lengths_.size() == batch_size, "target_lengths needs to have size to match batch_size");

  std::vector<int> input_lengths(input_lengths_.begin(), input_lengths_.end());
  std::vector<int> target_lengths(target_lengths_.begin(), target_lengths_.end());

  setCuDNNStreamToCurrent();
  TORCH_CHECK(BLANK == 0, "blank must be label 0 for cudnn_ctc_loss");
  // checked in dispatch:
  // assert other conditions for cudnnCTCLoss: all label lengths <= 256
  // all input lengths = logprob.size(0)

  auto handle = getCudnnHandle();

  cudnnCTCLossAlgo_t algo = (deterministic ? CUDNN_CTC_LOSS_ALGO_DETERMINISTIC : CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC);

  Tensor probs;
  // before 7.6 passing probs was the only option, starting with 7.1 the log probs/unnormalized log probs is the default
  if (cudnnGetVersion() < 7600) {
    probs = log_probs->softmax(2);
  } else {
    probs = *log_probs;
  }
  TensorDescriptor probs_desc{probs};
  Tensor grad = at::empty_like(probs);
  TensorDescriptor grad_desc{grad};

  CTCLossDescriptor ctc_loss_desc;
  ctc_loss_desc.set(CUDNN_DATA_FLOAT);

  size_t workspace_size;
  AT_CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize(handle, probs_desc.desc(), grad_desc.desc(),
                                              targets->data_ptr<int>(), target_lengths.data(), input_lengths.data(),
                                              algo, ctc_loss_desc.desc(), &workspace_size));


  Tensor workspace = at::empty(workspace_size, log_probs->options().dtype(kByte));
  Tensor costs = at::empty({log_probs->size(1)}, log_probs->options());

  AT_CUDNN_CHECK(cudnnCTCLoss(handle, probs_desc.desc(), probs.data_ptr(),
                              targets->data_ptr<int>(), target_lengths.data(), input_lengths.data(),
                              costs.data_ptr(), grad_desc.desc(), grad.data_ptr(), algo,
                              ctc_loss_desc.desc(), workspace.data_ptr(), workspace_size));

  return std::make_tuple(costs, grad);
}


}}  // namespace at::native

#endif
