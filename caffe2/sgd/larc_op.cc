#include "caffe2/sgd/larc_op.h"
#include <math.h>
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
void LarcOp<float, CPUContext>::Compute(
    TIndex N,
    const float* X_data,
    const float* dX_data,
    const float* wd,
    const float* lr_max,
    float offset,
    float trust,
    float lr_min,
    float* lr_rescaled) {
  float val = 1.0;
  float X_norm =
      sqrtf((ConstEigenVectorMap<float>(X_data, N).array()).square().sum());
  if (X_norm > 0) {
    float dX_norm =
        sqrtf((ConstEigenVectorMap<float>(dX_data, N).array()).square().sum());
    val = trust / (dX_norm / X_norm + (*wd) + offset);
  }
  val = fmin(val, *lr_max);
  val = fmax(val, lr_min);
  *lr_rescaled = val;
}

REGISTER_CPU_OPERATOR(Larc, LarcOp<float, CPUContext>);

OPERATOR_SCHEMA(Larc)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Implement Layer-wise Adaptive Rate Scaling Control (LARC). Before adding weight
decay, given a parameter tensor X and its gradient dX, the local learning rate
for X will be

local_lr = trust * norm(X) / ( norm(dX) + wd * norm(X) + offset * norm(X) )

      = trust / ( norm(dX) / norm(X) + wd + offset ),

where offset is a preset hyper-parameter to avoid numerical issue and trust
indicates how much we trust the layer to change its parameters during one update.
In this implementation, we uses l2 norm and the computed local learning rate is
clipped based on the upper bound lr_max and the lower bound lr_min:

local_lr = min(local_lr, lr_max) and local_lr = max(local_lr, lr_min)

)DOC")
    .Input(0, "X", "Parameter tensor")
    .Input(1, "dX", "Gradient tensor")
    .Input(2, "wd", "Weight decay")
    .Input(3, "lr_max)", "Upper bound of learning rate")
    .Output(0, "lr_rescaled", "Rescaled local learning rate")
    .Arg("offset", "rescaling offset parameter")
    .Arg("trust", "trust of the layer to change its weights")
    .Arg("lr_min", "minimum learning rate for clipping");

SHOULD_NOT_DO_GRADIENT(Larc);
} // namespace caffe2
