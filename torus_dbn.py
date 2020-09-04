import sys

import numpy as np
import pyro
import pyro.distributions as dist
import tqdm
from pyro import poutine
from pyro.infer import SVI, JitTraceEnum_ELBO
from pyro.infer.autoguide import init_to_sample, AutoDelta
from pyro.ops.indexing import Vindex
from pyro.util import ignore_jit_warnings
from pyro.optim import Adam

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from protein_parser import ProteinParser


def model(phis, psis, lengths, num_states=55,
          prior_conc=0.1, prior_loc=0.0):
    # From https://pyro.ai/examples/hmm.html
    with ignore_jit_warnings():
        num_sequences, max_length = map(int, phis.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    transition_probs = pyro.sample('transition_probs',
                                   dist.Dirichlet(torch.ones(num_states, num_states, dtype=torch.float)
                                                  * num_states)
                                   .to_event(1))
    phi_locs = pyro.sample('phi_locs',
                           dist.VonMises(torch.ones(num_states, dtype=torch.float) * prior_loc,
                                         torch.ones(num_states, dtype=torch.float) * prior_conc).to_event(1))
    psi_locs = pyro.sample('psi_locs',
                           dist.VonMises(torch.ones(num_states, dtype=torch.float) * prior_loc,
                                         torch.ones(num_states, dtype=torch.float) * prior_conc).to_event(1))
    element_plate = pyro.plate('elements', 1, dim=-1)
    with pyro.plate('sequences', num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        state = 0
        for t in pyro.markov(range(max_length)):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                state = pyro.sample(f'state_{t}', dist.Categorical(transition_probs[state]),
                                    infer={'enumerate': 'parallel'})
                phi = Vindex(phis)[batch, t]
                psi = Vindex(psis)[batch, t]
                with element_plate:
                    pyro.sample(f'phi_{t}', dist.VonMises(phi_locs[state], 1.0), obs=phi.unsqueeze(-1))
                    pyro.sample(f'psi_{t}', dist.VonMises(psi_locs[state], 1.0), obs=psi.unsqueeze(-1))


def main(_argv):
    aas, ds, phis, psis, lengths = ProteinParser.parsef_tensor('data/TorusDBN/top500.txt')
    guide = AutoDelta(poutine.block(model, hide_fn=lambda site: site['name'].startswith('state')),
                      init_loc_fn=init_to_sample)
    svi = SVI(model, guide, Adam(dict(lr=0.1)), JitTraceEnum_ELBO())
    total = 1000
    dataset = TensorDataset(phis, psis, lengths)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with tqdm.trange(total) as t:
        total_loss = float('inf')
        for i in t:
            losses = []
            num_batches = 0
            for j, (phis, psis, lengths) in enumerate(dataloader):
                loss = svi.step(phis, psis, lengths, num_states=5)
                print("BLA")
                losses.append(loss)
                num_batches += 1
                t.set_description_str(f"SVI ({j}:{i}:{total}): {loss:.2} [{total_loss:.2}]")
            total_loss = np.sum(losses) / (batch_size * num_batches)
    median = guide.median()
    print("Transition probs: ", median['transition_probs'].detach().numpy())
    print("Emission probs: ", median['emission_probs'].squeeze().detach().numpy())


if __name__ == '__main__':
    main(sys.argv)
