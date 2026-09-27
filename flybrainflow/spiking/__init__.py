"""M1 Step 2 -- the spiking simulator scaffold (see docs/M1_PLAN.md's own Step 2), plus Step 3's
pre-work pieces (sign resolution, the calibration control's shuffle, cell-type lookup).

Three Step 2 pieces, kept separate on purpose so each is testable on its own against a small
synthetic connectome (the CI-tier "logic" tests docs/M1_PLAN.md calls for), independent of ever
touching the real, Kaggle-only, ~164,740-neuron MaleCNS connectome:

  - `connectivity` -- turns a raw (pre, post, weight) edge list into one sparse weight matrix and a
    stable neuron-id <-> index mapping. Doesn't know or care where the edges came from.
  - `dynamics` -- the actual leaky-integrate-and-fire (LIF) update rule: one integer-arithmetic
    step of the whole population's membrane potential, given the previous tick's spikes and this
    tick's external input. Doesn't know or care how many neurons there are, or how many flies are
    being run at once -- everything is shaped `(n_flies, n_neurons)` from the start so batching
    more flies later is a size change, not a redesign.
  - `simulator` -- ties the two together into `SpikingSimulator`, the thing that actually owns
    state across ticks for one or more flies and exposes `.step(external_input) -> spikes`.

Three more pieces, added 2026-09-27 as Step 3 pre-work (found missing by that day's review pass --
see docs/M1_PLAN.md's Step 3 section for why each is needed before the calibration gate can run):

  - `neurotransmitters` -- resolves excitatory/inhibitory sign per neuron from predicted
    neurotransmitter identity, using the same convention Shiu et al.'s own model used (GABA/
    glutamate -> inhibitory, everything else -> excitatory), and applies it to raw edge weights.
  - `shuffle` -- the calibration gate's degree-preserving shuffled control: randomizes edges while
    keeping every node's in/out-degree and the graph's overall weight multiset exactly preserved.
  - `celltypes` -- cell-type name -> body-id lookup against a downloaded annotation table, for
    "stimulate sugar-GRNs, read out MN9" and Step 4's real sensory/motor wiring alike.

`ConnectomeBrain` (Step 4 -- wiring this into `Sim`/`Population`) is deliberately NOT working code
yet: per this file's own Step 4 text, real content is gated behind Step 3's calibration gate actually
passing, since its key parameters (sensory-injection scaling, the clock ratio) are meant to come
from that result, not be guessed in advance. An interface SKELETON exists at
`flybrainflow/brains/connectome.py` (added 2026-09-27 night) -- same method shape as `ToyBrain`,
every method raising `NotImplementedError`, every calibration-dependent constructor argument
required with no default -- so the eventual real work is "fill this in" rather than "start from
nothing." It is not imported here or from `flybrainflow.brains`'s own `__init__.py` (see that
module's docstring): it needs torch, and `flybrainflow.brains` otherwise doesn't. See
docs/M1_PLAN.md's Step 3/4 sections and docs/JOURNAL.md's 2026-09-27 entries.
"""

from __future__ import annotations

from .celltypes import body_ids_for_type
from .connectivity import Connectivity, load_connectivity
from .dynamics import LIFParams, LIFState, step
from .neurotransmitters import SIGN_CONVENTION, apply_signs, resolve_signs
from .shuffle import shuffle_connectivity
from .simulator import SpikingSimulator

__all__ = [
    "Connectivity",
    "load_connectivity",
    "LIFParams",
    "LIFState",
    "step",
    "SpikingSimulator",
    "SIGN_CONVENTION",
    "resolve_signs",
    "apply_signs",
    "shuffle_connectivity",
    "body_ids_for_type",
]
