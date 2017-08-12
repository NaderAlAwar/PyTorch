#include "caffe2/operators/boolean_mask_ops.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace {

template <class Context>
class BooleanMaskLengthsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BooleanMaskLengthsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& lengths = Input(0);
    auto& mask = Input(1);
    auto* lengthsOut = Output(0);
    CAFFE_ENFORCE(lengths.ndim() == 1);
    CAFFE_ENFORCE(mask.ndim() == 1);
    const auto* lengthsPtr = lengths.template data<T>();
    const auto* maskPtr = mask.template data<bool>();
    auto totalLength =
        std::accumulate(lengthsPtr, lengthsPtr + lengths.size(), 0);
    CAFFE_ENFORCE(mask.size() == totalLength);
    lengthsOut->ResizeLike(lengths);
    auto* lengthsOutPtr = lengthsOut->template mutable_data<T>();
    int p = 0;
    for (int i = 0; i < lengths.size(); ++i) {
      T lengthOut = 0;
      for (int j = 0; j < lengthsPtr[i]; ++j) {
        if (maskPtr[p++]) {
          ++lengthOut;
        }
      }
      lengthsOutPtr[i] = lengthOut;
    }
    return true;
  }
};
} // namespace

template <>
bool BooleanMaskOp<CPUContext>::RunOnDevice() {
  auto& data = Input(0);
  auto& mask = Input(1);
  auto* dataOut = Output(0);
  CAFFE_ENFORCE(data.ndim() >= 1);
  CAFFE_ENFORCE_EQ(mask.ndim(), 1);
  CAFFE_ENFORCE(data.dims()[0] == mask.dims()[0]);

  const auto* maskPtr = mask.template data<bool>();
  int numOutputs = 0;
  int outerSize = mask.size();
  for (int i = 0; i < outerSize; ++i) {
    if (maskPtr[i]) {
      ++numOutputs;
    }
  }
  std::vector<TIndex> outShape;
  outShape.push_back(numOutputs);
  outShape.insert(outShape.end(), data.dims().begin() + 1, data.dims().end());
  dataOut->Resize(outShape);
  auto* outPtr = (char*)dataOut->raw_mutable_data(data.meta());

  int64_t* out_vec;
  if (OutputSize() == 2) {
    auto* indicesOut = Output(1);
    indicesOut->Resize(numOutputs);
    out_vec = indicesOut->template mutable_data<int64_t>();
  }

  if (numOutputs == 0) {
    return true;
  }
  const auto innerSize = data.size_from_dim(1);
  const auto innerSizeBytes = innerSize * data.meta().itemsize();

  TIndex lastStart = -1;
  const auto* inPtr = (char*)data.raw_data();
  TIndex outStart = 0;

  for (TIndex i = 0;; ++i) {
    // mask was true and either a) became false, or b) sequence finished
    if (lastStart != -1 && ((i >= outerSize) || !maskPtr[i])) {
      const auto* src = inPtr + lastStart * innerSizeBytes;
      auto* dst = outPtr + outStart * innerSizeBytes;
      int numItems = i - lastStart;
      context_.template CopyItems<CPUContext, CPUContext>(
          data.meta(), numItems * innerSize, src, dst);
      outStart += numItems;
      lastStart = -1;
    }
    if (i >= outerSize) {
      break;
    }
    // mask was false and became true
    if (lastStart == -1 && maskPtr[i]) {
      lastStart = i;
    }
    if (maskPtr[i] && OutputSize() == 2) {
      *(out_vec++) = i;
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(BooleanMask, BooleanMaskOp<CPUContext>);
REGISTER_CPU_OPERATOR(BooleanMaskLengths, BooleanMaskLengthsOp<CPUContext>);

OPERATOR_SCHEMA(BooleanMask)
    .NumInputs(2)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(
Given a data tensor and a 1D boolean mask tensor, returns a tensor containing
only the elements corresponding to positions where the mask is true.
)DOC")
    .Input(0, "data", "The 1D, original data tensor.")
    .Input(1, "mask", "A tensor of bools of same shape as `data`.")
    .Output(0, "masked_data", "A tensor of same type as `data`.")
    .Output(1, "masked_indices", "A tensor for indices.");

OPERATOR_SCHEMA(BooleanMaskLengths)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a tensor of int32 segment lengths and a mask (boolean) tensor, return
the segment lengths of a corresponding segmented tensor after BooleanMask is
applied.
)DOC")
    .Input(0, "lengths", "A 1D int32 tensor representing segment lengths.")
    .Input(1, "mask", "A 1D bool tensor of values to keep.")
    .Output(0, "masked_lengths", "Segment lengths of a masked tensor.");

NO_GRADIENT(BooleanMask)
NO_GRADIENT(BooleanMaskLengths);

const float minf = -1.0f * std::numeric_limits<float>::infinity();

// Template this on a functor object so we can generate different
// implementations at compile time and have a better chance of inlining
template <typename Functor>
void MaskWithFunctor(
    size_t N,
    size_t D,
    const float* in,
    Functor fn,
    float* out) {
  // TODO(T20952436): vector implementation
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      auto val = in[D * i + j];
      out[D * i + j] = (fn(i, j, val) ? minf : val);
    }
  }
}

namespace {

class SequenceFunctor {
 public:
  explicit SequenceFunctor(const int* sl) : sl(sl) {}
  bool operator()(int i, int j, float /* val*/) {
    return j >= sl[i];
  }

 private:
  const int* sl;
};

class UpperFunctor {
 public:
  bool operator()(int i, int j, float /* val */) {
    return j > i;
  }
};

class LowerFunctor {
 public:
  bool operator()(int i, int j, float /* val */) {
    return j < i;
  }
};

class UpperDiagFunctor {
 public:
  bool operator()(int i, int j, float /* val */) {
    return j >= i;
  }
};

class LowerDiagFunctor {
 public:
  bool operator()(int i, int j, float /* val */) {
    return j <= i;
  }
};

} // namespace

template <>
bool SequenceMaskOp<CPUContext>::RunOnDevice() {
  const Tensor<CPUContext>* input = &Input(0);
  const Tensor<CPUContext>* sequence_lengths = nullptr;

  if (mode_ == "sequence") {
    sequence_lengths = &Input(1);
  }

  auto* output = Output(0);
  output->ResizeLike(*input);

  const auto canonical_axis = input->canonical_axis_index(axis_);
  const int left = input->size_to_dim(canonical_axis);
  const int right = input->size_from_dim(canonical_axis);

  if (mode_ == "sequence") {
    CAFFE_ENFORCE(
        sequence_lengths, "Sequence length not provided for mode 'sequence'!");
    MaskWithFunctor(
        left,
        right,
        input->data<float>(),
        SequenceFunctor(sequence_lengths->data<int>()),
        output->mutable_data<float>());
  } else if (mode_ == "upper") {
    MaskWithFunctor(
        left,
        right,
        input->data<float>(),
        UpperFunctor(),
        output->mutable_data<float>());
  } else if (mode_ == "lower") {
    MaskWithFunctor(
        left,
        right,
        input->data<float>(),
        LowerFunctor(),
        output->mutable_data<float>());
  } else if (mode_ == "upperdiag") {
    MaskWithFunctor(
        left,
        right,
        input->data<float>(),
        UpperDiagFunctor(),
        output->mutable_data<float>());
  } else if (mode_ == "lowerdiag") {
    MaskWithFunctor(
        left,
        right,
        input->data<float>(),
        LowerDiagFunctor(),
        output->mutable_data<float>());
  } else {
    CAFFE_ENFORCE(false, "Unsupported mode for SequenceMaskOp!");
    return false;
  }

  return true;
}

REGISTER_CPU_OPERATOR(SequenceMask, SequenceMaskOp<CPUContext>);

OPERATOR_SCHEMA(SequenceMask)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Mask op designed for use in attention mechanisms for sequence modeling tasks.

Two current operating modes:
1) Given a 2D input tensor and 1D tensor of sequence lengths, for each row i in
      the input tensor, set elements in that row to -inf if their column index
      j >= sequence_lengths[i]. This mode takes two inputs and argument mode =
      'sequence'
2) Triangular mask. Given row index i and column index j, set elements to -inf
      given the following conditions:

      mode='upper', x_ij = -inf if j < i
      mode='lower', x_ij = -inf if j > i
      mode='upperdiag', x_ij = -inf if j <= i
      mode='lowerdiag', x_ij = -inf if j >= i

    This mode takes one input.
)DOC")
    .Input(0, "input", "Tensor to apply masking to")
    .Input(1, "sequence_lengths", "1D Tensor of sequence lengths for mode #1")
    .Output(0, "masked_tensor", "Input tensor with masking applied")
    .Arg(
        "mode",
        "(string) Mode selection. Possible values: "
        "'sequence', 'upper', 'lower', 'upperdiag', 'lowerdiag'")
    .Arg(
        "axis",
        "(int) Beginning axis of row elements. All dimensions to the left "
        "will be treated as row indices and those to the right (inclusive) "
        "will be treated as column indices in the 2D mask");

} // namespace caffe2
