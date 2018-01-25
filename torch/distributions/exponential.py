from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all, lazy_property


class Exponential(ExponentialFamily):
    r"""
    Creates a Exponential distribution parameterized by `rate`.

    Example::

        >>> m = Exponential(torch.Tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        rate (float or Tensor or Variable): rate = 1 / scale of the distribution
    """
    params = {'rate': constraints.positive}
    support = constraints.positive
    has_rsample = True
    _zero_carrier_measure = True

    def __init__(self, rate):
        self.rate, = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, Number) else self.rate.size()
        super(Exponential, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.rate.new(*shape).exponential_() / self.rate

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        return self.rate.log() - self.rate * value

    def entropy(self):
        return 1.0 - torch.log(self.rate)

    def natural_params(self):
        return self._natural_params

    @lazy_property
    def _natural_params(self):
        try:
            V1 = Variable(-self.rate, requires_grad=True)
        except:
            V1 = Variable(-self.rate.data, requires_grad=True)
        return (V1, )

    def log_normalizer(self):
        x, = self._natural_params
        return -torch.log(-x)
