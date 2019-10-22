#pragma once

#include <torch/nn/functional/pooling.h>
#include <torch/nn/options/upsampling.h>

#include <cmath>

namespace torch {
namespace nn {
namespace functional {

inline Tensor interpolate(const Tensor& input, const InterpolationOptions& options) {
  auto _check_size_scale_factor = [input, options](size_t dim) {
    if (options.size() == c10::nullopt &&
        options.scale_factor() == c10::nullopt) {
      TORCH_CHECK(false, "either size or scale_factor should be defined");
    }
    if (options.size() != c10::nullopt &&
        options.scale_factor() != c10::nullopt) {
      TORCH_CHECK(false, "only one of size or scale_factor should be defined");
    }
    if (options.scale_factor() != c10::nullopt &&
        options.scale_factor()->size() != dim) {
      TORCH_CHECK(
          false,
          "scale_factor shape must match input shape. "
          "Input is ", input.dim(), "D, scale_factor size is ",
          options.scale_factor()->size());
    }
  };

  auto _output_size = [input, options, _check_size_scale_factor](size_t dim) {
    _check_size_scale_factor(dim);
    if (options.size() != c10::nullopt) {
      return *options.size();
    }
    auto scale_factors = *options.scale_factor();

    std::vector<int64_t> sizes;
    for (size_t i = 0; i < dim; ++i) {
      sizes.push_back(static_cast<int64_t>(std::floor(
          static_cast<double>(input.size(i + 2)) * scale_factors[i])));
    }
    return sizes;
  };

  if (options.mode() == Interpolation::Nearest ||
      options.mode() == Interpolation::Area) {
    if (options.align_corners() != c10::nullopt) {
      TORCH_CHECK(
          false,
          "align_corners option can only be set with the "
          "interpolating modes: linear | bilinear | bicubic | trilinear");
    }
  } else {
    if (options.align_corners() == c10::nullopt) {
      TORCH_WARN(
          "Default upsampling behavior when mode is linear, bilinear, bicubic, "
          "or trilinear, has changed to align_corners=False since 0.4.0. "
          "Please specify align_corners=True if the old behavior is desired. "
          "See the documentation of nn.Upsample for details.");
    }
  }
  bool align_corners = options.align_corners().value_or(false);

  if (input.dim() == 3 && options.mode() == Interpolation::Nearest) {
    return torch::upsample_nearest1d(input, _output_size(1));
  } else if (input.dim() == 4 && options.mode() == Interpolation::Nearest) {
    return torch::upsample_nearest2d(input, _output_size(2));
  } else if (input.dim() == 5 && options.mode() == Interpolation::Nearest) {
    return torch::upsample_nearest3d(input, _output_size(3));
  } else if (input.dim() == 3 && options.mode() == Interpolation::Area) {
    return adaptive_avg_pool1d(input, _output_size(1));
  } else if (input.dim() == 4 && options.mode() == Interpolation::Area) {
    return adaptive_avg_pool2d(input, _output_size(2));
  } else if (input.dim() == 5 && options.mode() == Interpolation::Area) {
    return adaptive_avg_pool3d(input, _output_size(3));
  } else if (input.dim() == 3 && options.mode() == Interpolation::Linear) {
    return torch::upsample_linear1d(input, _output_size(1), align_corners);
  } else if (input.dim() == 3 && options.mode() == Interpolation::Bilinear) {
    TORCH_CHECK(false, "Got 3D input, but bilinear mode needs 4D input");
  } else if (input.dim() == 3 && options.mode() == Interpolation::Trilinear) {
    TORCH_CHECK(false, "Got 3D input, but trilinear mode needs 5D input");
  } else if (input.dim() == 4 && options.mode() == Interpolation::Linear) {
    TORCH_CHECK(false, "Got 4D input, but linear mode needs 3D input");
  } else if (input.dim() == 4 && options.mode() == Interpolation::Bilinear) {
    return torch::upsample_bilinear2d(input, _output_size(2), align_corners);
  } else if (input.dim() == 4 && options.mode() == Interpolation::Trilinear) {
    TORCH_CHECK(false, "Got 4D input, but trilinear mode needs 5D input");
  } else if (input.dim() == 5 && options.mode() == Interpolation::Linear) {
    TORCH_CHECK(false, "Got 5D input, but linear mode needs 3D input");
  } else if (input.dim() == 5 && options.mode() == Interpolation::Bilinear) {
    TORCH_CHECK(false, "Got 5D input, but bilinear mode needs 4D input");
  } else if (input.dim() == 5 && options.mode() == Interpolation::Trilinear) {
    return torch::upsample_trilinear3d(input, _output_size(3), align_corners);
  } else if (input.dim() == 4 && options.mode() == Interpolation::Bicubic) {
    return torch::upsample_bicubic2d(input, _output_size(2), align_corners);
  } else {
    TORCH_CHECK(
        false,
        "Input Error: Only 3D, 4D and 5D input Tensors supported "
        "(got ", input.dim(), "D) for the modes: nearest | linear | "
        "bilinear | bicubic | trilinear");
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
