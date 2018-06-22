#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>

#include <ATen/ATen.h>

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace optim {

struct RMSpropOptions {
  RMSpropOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, alpha) = 0.99;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(bool, centered) = false;
};

class RMSprop : public Optimizer {
 public:
  RMSprop(std::shared_ptr<nn::Module> model, const RMSpropOptions& options);

  template <typename ModuleType>
  RMSprop(
      nn::ModuleHolder<ModuleType> module_holder,
      const RMSpropOptions& options)
      : RMSprop(module_holder.get(), options) {}

  template <typename ParameterContainer>
  explicit RMSprop(
      ParameterContainer&& parameters,
      const RMSpropOptions& options)
      : Optimizer(std::move(parameters)),
        options_(options),
        square_average_buffers_(detail::zeros_like(parameters_)) {
    if (options_.momentum_ > 0) {
      momentum_buffers_ = detail::zeros_like(parameters_);
    }
    if (options_.centered_ > 0) {
      grad_average_buffers_ = detail::zeros_like(parameters_);
    }
  }

  void step() override;

  const RMSpropOptions& options() const noexcept;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(square_average_buffers_));
    ar(CEREAL_NVP(momentum_buffers_));
    ar(CEREAL_NVP(grad_average_buffers_));
  }

 private:
  friend class cereal::access;
  RMSprop() : options_(0) {}

  RMSpropOptions options_;

  std::vector<Variable> square_average_buffers_;
  std::vector<Variable> momentum_buffers_;
  std::vector<Variable> grad_average_buffers_;
};

} // namespace optim
} // namespace torch
