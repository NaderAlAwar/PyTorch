#include "caffe2/operators/elu_op.h"

#include "caffe2/operators/activation_ops_cudnn.h"

namespace caffe2 {

template <>
class CuDNNActivationOp<CUDNN_ACTIVATION_ELU> final
    : public CuDNNActivationOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNActivationOp(const OperatorDef& operator_def, Workspace* ws)
      : CuDNNActivationOpBase(operator_def, ws),
        OP_SINGLE_ARG(float, "alpha", alpha_, 1.0f) {
    CUDNN_ENFORCE(cudnnSetActivationDescriptor(
        act_desc_,
        CUDNN_ACTIVATION_ELU,
        CUDNN_PROPAGATE_NAN,
        static_cast<double>(alpha_)));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    if (X.size() == 0) {
      Y->template mutable_data<T>();
      return true;
    }
    this->SetTensorDescriptor(cudnnTypeWrapper<T>::type, X.size());
    CUDNN_ENFORCE(cudnnActivationForward(
        this->cudnn_wrapper_.inline_cudnn_handle(),
        this->act_desc_,
        cudnnTypeWrapper<T>::kOne(),
        this->data_desc_,
        X.template data<T>(),
        cudnnTypeWrapper<T>::kZero(),
        this->data_desc_,
        Y->template mutable_data<T>()));
    return true;
  }

 private:
  const float alpha_;
};

template <>
class CuDNNActivationGradientOp<CUDNN_ACTIVATION_ELU> final
    : public CuDNNActivationOpBase {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  CuDNNActivationGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CuDNNActivationOpBase(operator_def, ws),
        OP_SINGLE_ARG(float, "alpha", alpha_, 1.0f) {
    CUDNN_ENFORCE(cudnnSetActivationDescriptor(
        act_desc_,
        CUDNN_ACTIVATION_ELU,
        CUDNN_PROPAGATE_NAN,
        static_cast<double>(alpha_)));
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& Y = Input(0);
    const auto& dY = Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(Y);
    if (Y.size() == 0) {
      dX->template mutable_data<T>();
      return true;
    }
    this->SetTensorDescriptor(cudnnTypeWrapper<T>::type, Y.size());
    CUDNN_ENFORCE(cudnnActivationBackward(
        this->cudnn_wrapper_.inline_cudnn_handle(),
        this->act_desc_,
        cudnnTypeWrapper<T>::kOne(),
        this->data_desc_,
        Y.template data<T>(),
        this->data_desc_,
        dY.template data<T>(),
        this->data_desc_,
        Y.template data<T>(), // Use Y_data as placeholder here.
        cudnnTypeWrapper<T>::kZero(),
        this->data_desc_,
        dX->template mutable_data<T>()));
    return true;
  }

 private:
  const float alpha_;
};

REGISTER_CUDNN_OPERATOR(Elu, CuDNNActivationOp<CUDNN_ACTIVATION_ELU>);
REGISTER_CUDNN_OPERATOR(
    EluGradient,
    CuDNNActivationGradientOp<CUDNN_ACTIVATION_ELU>);

} // namespace caffe2
