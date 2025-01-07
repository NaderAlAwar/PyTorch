import math
from typing import Optional, Union
from typing_extensions import Self

import torch
from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.cauchy import Cauchy
from torch.distributions.constraints import Constraint
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform
from torch.types import _size


__all__ = ["HalfCauchy"]


class HalfCauchy(TransformedDistribution):
    r"""
    Creates a half-Cauchy distribution parameterized by `scale` where::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # half-cauchy distributed with scale=1
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): scale of the full Cauchy distribution
    """
    arg_constraints: dict[str, Constraint] = {"scale": constraints.positive}
    support = constraints.nonnegative  # type: ignore[assignment]
    has_rsample: bool = True
    base_dist: Cauchy

    def __init__(
        self,
        scale: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        base_dist = Cauchy(0, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape: _size, _instance: Optional[Self] = None) -> Self:
        new = self._get_checked_instance(HalfCauchy, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        return torch.full(
            self._extended_shape(),
            math.inf,
            dtype=self.scale.dtype,
            device=self.scale.device,
        )

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.scale)

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        value = torch.as_tensor(
            value, dtype=self.base_dist.scale.dtype, device=self.base_dist.scale.device
        )
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob = torch.where(value >= 0, log_prob, -inf)
        return log_prob

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob: Tensor) -> Tensor:
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self) -> Tensor:
        return self.base_dist.entropy() - math.log(2)
