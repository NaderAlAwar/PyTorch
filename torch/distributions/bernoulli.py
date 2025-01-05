## mypy: allow-untyped-defs
from numbers import Number
from typing import Any, Optional
from typing_extensions import Self

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.types import _size


__all__ = ["Bernoulli"]


class Bernoulli(ExponentialFamily):
    r"""
    Creates a Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Bernoulli(torch.tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
        tensor([ 0.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
    """
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.boolean
    has_enumerate_support = True
    _mean_carrier_measure = 0

    def __init__(
        self,
        probs: Optional[Tensor | Number] = None,
        logits: Optional[Tensor | Number] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, Number)
            (self.probs,) = broadcast_all(probs)
        else:
            is_scalar = isinstance(logits, Number)
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape: _size, _instance: Optional[Self] = None) -> Self:
        new = self._get_checked_instance(Bernoulli, _instance)
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Bernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args: Any, **kwargs: Any) -> Tensor:
        return self._param.new(*args, **kwargs)

    @property
    def mean(self) -> Tensor:
        return self.probs

    @property
    def mode(self) -> Tensor:
        mode = (self.probs >= 0.5).to(self.probs)
        mode[self.probs == 0.5] = nan
        return mode

    @property
    def variance(self) -> Tensor:
        return self.probs * (1 - self.probs)

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self) -> torch.Size:
        return self._param.size()

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.bernoulli(self.probs.expand(shape))

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return -binary_cross_entropy_with_logits(logits, value, reduction="none")

    def entropy(self) -> Tensor:
        return binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction="none"
        )

    def enumerate_support(self, expand: bool = True) -> Tensor:
        values = torch.arange(2, dtype=self._param.dtype, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (torch.logit(self.probs),)

    def _log_normalizer(self, x: Tensor) -> Tensor:
        return torch.log1p(torch.exp(x))
