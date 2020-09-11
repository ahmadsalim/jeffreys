import sys

import jax
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import tqdm
from jax.random import PRNGKey
from numpyro.contrib.funsor import log_density, enum, config_enumerate
from numpyro.handlers import seed, trace
from numpyro.infer import SVI, ELBO, NUTS, MCMC, RenyiELBO
from numpyro.infer.util import _guess_max_plate_nesting
from numpyro.optim import Adam
from scipy import stats
import numpy as np


def gmm(data, num_components=3):
    mus = numpyro.sample('mus', dist.Normal(jnp.zeros(num_components),
                                            jnp.ones(num_components) * 100.).to_event(1))
    sigmas = numpyro.sample('sigmas', dist.HalfNormal(jnp.ones(num_components) * 100.).to_event(1))
    mixture_probs = numpyro.sample('mixture_probs', dist.Dirichlet(
        jnp.ones(num_components) / num_components))
    with numpyro.plate('data', len(data), dim=-1):
        z = numpyro.sample('z', dist.Categorical(mixture_probs))
        numpyro.sample('ll', dist.Normal(mus[z], sigmas[z]), obs=data)


def gmm_guide(data, num_components=3):
    mus_val = numpyro.param('mus_val', jnp.array(stats.norm.rvs(size=num_components) * 1000),
                            constraint=dist.constraints.real)
    sigmas_val = numpyro.param('sigmas_val', jnp.ones(num_components), constraint=dist.constraints.positive)
    mus = numpyro.sample('mus', dist.Delta(mus_val))
    sigmas = numpyro.sample('sigmas', dist.Delta(sigmas_val))
    mixture_probs_val = numpyro.param('mixture_probs_val',
                                      jax.nn.softmax(stats.norm.rvs(size=num_components)),
                                      constraint=dist.constraints.simplex)
    mixture_probs = numpyro.sample('mixture_probs', dist.Delta(mixture_probs_val))


def generate_data(num_samples=1000):
    zs = stats.multinomial.rvs(1, (0.5, 0.35, 0.15), num_samples).argmax(-1)
    mus = np.array([-100., 0., 34.])
    sigmas = np.array([10., 1., 0.3])
    return stats.norm.rvs(loc=mus[zs], scale=sigmas[zs], size=num_samples)


def main(_args):
    data = generate_data()
    init_rng_key = PRNGKey(1273)
    # nuts = NUTS(gmm)
    # mcmc = MCMC(nuts, 100, 1000)
    # mcmc.print_summary()
    seeded_gmm = seed(gmm, init_rng_key)
    model_trace = trace(seeded_gmm).get_trace(data)
    max_plate_nesting = _guess_max_plate_nesting(model_trace)
    enum_gmm = enum(config_enumerate(gmm), - max_plate_nesting - 1)
    svi = SVI(enum_gmm, gmm_guide, Adam(0.1), RenyiELBO(-10.))
    svi_state = svi.init(init_rng_key, data)
    upd_fun = jax.jit(svi.update)
    with tqdm.trange(100_000) as pbar:
        for i in pbar:
            svi_state, loss = upd_fun(svi_state, data)
            pbar.set_description(f"SVI {loss}", True)
    print(svi.get_params(svi_state))


if __name__ == '__main__':
    main(sys.argv)
