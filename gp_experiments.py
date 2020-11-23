import sys
from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.pyplot as plt
import torch
from pyro import distributions as dist


class GPKernel(ABC):
    @abstractmethod
    def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OUKernel(GPKernel):
    def __init__(self, drift, scale):
        self.drift = drift
        self.scale = scale

    def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        diff = points1.unsqueeze(1) - points2.unsqueeze(0)
        absdiff = diff.abs().sum(-1)
        return (- self.drift * absdiff).exp() * self.scale ** 2 / (2 * self.drift)


# Based on http://www.auai.org/uai2013/prints/papers/244.pdf and
# http://docs.pyro.ai/en/stable/contrib.gp.html
class VSGP(dist.Distribution):
    """
    Variational Sparse Gaussian Process
    """

    def __init__(self, kernel: GPKernel, inducing_set: torch.Tensor,
                 output_distribution_f: Callable[[torch.Tensor, torch.Tensor], dist.Distribution],
                 *, input_data: torch.Tensor, eps=1e-7):
        super().__init__()
        self.kernel = kernel
        self.inducing_set = inducing_set
        self.output_distribution_f = output_distribution_f
        self.input_data = input_data
        self._eps = eps
        self._compute_out_dist()

    def _compute_out_dist(self):
        kmm = self.kernel.forward(self.inducing_set.unsqueeze(-1), self.inducing_set.unsqueeze(-1))
        kmm = kmm + torch.eye(kmm.size()[0]) * self._eps
        knm = self.kernel.forward(self.input_data, self.inducing_set.unsqueeze(-1))
        kmm_inv = kmm.inverse()
        kmn = knm.transpose(-1, -2)
        knn = self.kernel.forward(self.input_data, self.input_data)
        self._f_mean = knm @ kmm_inv @ self.inducing_set
        self._f_var = knn - knm @ kmm_inv @ kmn
        self._out_dist = self.output_distribution_f(self._f_mean, self._f_var)

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def conjugate_update(self, other):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        return self._out_dist.sample(*args, **kwargs)

    def log_prob(self, x, *args, **kwargs):
        return self._out_dist.log_prob(x, *args, **kwargs)


def main(_args):
    inducing_set = torch.randn(10) * 10
    kernel = OUKernel(5.3, 0.1)
    xs, ys = torch.meshgrid(torch.arange(0, 30.0), torch.arange(0, 30.0))
    inp = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)
    out = torch.sin(1e-5 * inp[..., 0]) / (inp[..., 1] + 1e-5)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(inp[..., 0].numpy(), inp[..., 1].numpy(), out)
    plt.show()
    vsgp = VSGP(kernel, inducing_set, dist.MultivariateNormal, input_data=inp)
    res = vsgp.sample(sample_shape=(30,))
    lp = vsgp.log_prob(out)
    x = 1


if __name__ == '__main__':
    main(sys.argv)
