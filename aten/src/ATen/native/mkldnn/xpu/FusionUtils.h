#pragma once
#include <detail/oneDNN.h>

namespace at::native::xpu {
at::native::onednn::Attr unary_attr_with_arg(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm,
    onednn::Attr attr);

at::native::onednn::Attr string_to_unary_attr(onednn::Attr attr);

at::native::onednn::Attr construct_unary_attr(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm,
    onednn::Attr attr);

at::native::onednn::Attr construct_binary_attr(
    c10::string_view binary,
    c10::optional<at::Scalar> alpha,
    const Tensor& other,
    onednn::Attr attr);

} // namespace at::native::xpu
