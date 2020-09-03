import sys

import pyro
import pyro.distributions as dist
import torch
import tqdm
from pyro import poutine
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta, init_to_sample
from pyro.ops.indexing import Vindex
from pyro.util import ignore_jit_warnings
from pyro.optim import Adam


def model(transition_alphas, emission_prior_loc, emission_prior_cov, lengths,
          sequences=None, batch_size=None):
    # From https://pyro.ai/examples/hmm.html
    with ignore_jit_warnings():
        if sequences is not None:
            num_sequences, max_length, data_dim = map(int, sequences.shape)
            assert lengths.shape == (num_sequences,)
            assert lengths.max() <= max_length
        else:
            data_dim = emission_prior_loc.size(-2)
            num_sequences = int(lengths.shape[0])
            max_length = int(lengths.max())
    transition_probs = pyro.sample('transition_probs',
                                   dist.Dirichlet(transition_alphas).to_event(1))
    emission_locs = pyro.sample('emission_locs',
                                dist.MultivariateNormal(emission_prior_loc, emission_prior_cov).to_event(1))
    element_plate = pyro.plate('elements', data_dim, dim=-1)
    with pyro.plate('sequences', num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        state = 0
        for t in pyro.markov(range(max_length)):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                state = pyro.sample(f'state_{t}', dist.Categorical(transition_probs[state]),
                                    infer={'enumerate': 'parallel'})
                obs_element = Vindex(sequences)[batch, t] if sequences is not None else None
                with element_plate:
                    element_loc = emission_locs[state.squeeze(-1)]
                    element = pyro.sample(f'element_{t}',
                                          dist.Normal(element_loc, torch.ones_like(element_loc)),
                                          obs=obs_element)


def main(_argv):
    transition_alphas = torch.tensor([[10., 90.],
                                      [90., 10.]])
    emission_prior_loc = torch.tensor([[30., 10.],
                                       [-50., 5.]])
    emission_prior_cov = torch.tensor([[[1., 0.],
                                        [0., 1.]],
                                       [[50., 12.],
                                        [12., 50]]])
    lengths = torch.randint(10, 30, (10000,))
    trace = poutine.trace(model).get_trace(transition_alphas, emission_prior_loc,
                                           emission_prior_cov, lengths)
    obs_sequences = [site['value'] for name, site in
                     trace.nodes.items() if name.startswith("element_")]
    obs_sequences = torch.stack(obs_sequences, dim=1)
    guide = AutoDelta(poutine.block(model, hide_fn=lambda site: site['name'].startswith('state')),
                      init_loc_fn=init_to_sample)
    svi = SVI(model, guide, Adam(dict(lr=0.1)), JitTraceEnum_ELBO())
    total = 1000
    with tqdm.trange(total) as t:
        for i in t:
            loss = svi.step(torch.ones((2, 2), dtype=torch.float) * 0.5,
                            torch.ones((2, 2), dtype=torch.float),
                            torch.diag_embed(torch.ones((2, 2), dtype=torch.float)) * 100,
                            lengths, obs_sequences)
            t.set_description_str(f"SVI ({i}/{total}): {loss}")
    median = guide.median()
    print("Transition probs: ", median['transition_probs'].detach().numpy())
    print("Emission locs: ", median['emission_locs'].squeeze().detach().numpy())


if __name__ == '__main__':
    main(sys.argv)
