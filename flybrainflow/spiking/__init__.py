"""M1 Step 2 -- the spiking simulator scaffold (see docs/M1_PLAN.md's own Step 2).

Three pieces, kept separate on purpose so each is testable on its own against a small synthetic
connectome (the CI-tier "logic" tests docs/M1_PLAN.md calls for), independent of ever touching the
real, Kaggle-only, ~164,740-neuron MaleCNS connectome:

  - `connectivity` -- turns a raw (pre, post, weight) edge list into one sparse weight matrix and a
    stable neuron-id <-> index mapping. Doesn't know or care where the edges came from.
  - `dynamics` -- the actual leaky-integrate-and-fire (LIF) update rule: one integer-arithmetic
    step of the whole population's membrane potential, given the previous tick's spikes and this
    tick's external input. Doesn't know or care how many neurons there are, or how many flies are
    being run at once -- everything is shaped `(n_flies, n_neurons)` from the start so batching
    more flies later is a size change, not a redesign.
  - `simulator` -- ties the two together into `SpikingSimulator`, the thing that actually owns
    state across ticks for one or more flies and exposes `.step(external_input) -> spikes`.

None of this is wired into `Sim`/`Population` yet -- that's M1 Step 4's job, once senses and motors
are wired to specific neurons. Step 2's own "done when" is narrower: this scaffold runs a full
connectome forward in time, on made-up/neutral input, without crashing or its activity blowing up.
"""

from __future__ import annotations

from .connectivity import Connectivity, load_connectivity
from .dynamics import LIFParams, LIFState, step
from .simulator import SpikingSimulator

__all__ = [
    "Connectivity",
    "load_connectivity",
    "LIFParams",
    "LIFState",
    "step",
    "SpikingSimulator",
]
