"""`SpikingSimulator` -- owns state across ticks for one or more flies running the same connectome,
and is the one thing outside this package that other code should ever import directly (`dynamics`
and `connectivity` are implementation details it composes).

Not wired into `Sim`/`Population` yet -- see this package's own `__init__.py` docstring. This is
the standalone scaffold M1 Step 2 asks for: something that can be pointed at a connectome and run
forward in time without crashing or its activity blowing up, on made-up input, before anything
about real senses or motors exists.
"""

from __future__ import annotations

import torch

from .connectivity import Connectivity
from .dynamics import LIFParams, LIFState, step


class SpikingSimulator:
    """Runs `n_flies` independent copies of the same connectome forward in time. "Independent"
    matters: each fly has its own `LIFState` (membrane potential, refractory counters, last spikes)
    but all flies share the one `Connectivity.weight_matrix` -- the "one shared weight matrix, many
    flies" pattern `docs/M1_PLAN.md`'s Step 2 calls out, already used by the M0 toy brain (see
    `ToyBrain`'s own docstring). Flies never interact with each other here; batching is purely a
    performance shape (one tensor op instead of a Python loop over flies), not a channel for one
    fly's activity to leak into another's.
    """

    def __init__(self, connectivity: Connectivity, params: LIFParams | None = None, *, n_flies: int = 1) -> None:
        """Runs on whatever device `connectivity.weight_matrix` is already on -- call
        `connectivity.to("cuda")` (see `Connectivity.to`) before constructing this if the weight
        matrix should live on a GPU; `SpikingSimulator` itself never decides that, it just follows.
        """
        self.connectivity = connectivity
        self.params = params or LIFParams()
        self.n_flies = n_flies
        self._device = connectivity.weight_matrix.device
        self.state = LIFState.zeros(n_flies, connectivity.n_neurons, self.params, device=self._device)
        self.t_step = 0  # count of internal spiking sub-steps run so far, for logging/debugging

    def reset(self) -> None:
        """Back to tick zero, same connectome and params. Flies keep whatever seed-derived
        behavior they had (there's no per-fly randomness in the base LIF update itself yet -- see
        `dynamics.py` -- so this is currently just a state zeroing; kept as its own method so a
        future noise/stochastic-input addition has one obvious place to also reset a per-fly RNG).
        """
        self.state = LIFState.zeros(self.n_flies, self.connectivity.n_neurons, self.params, device=self._device)
        self.t_step = 0

    def step(self, external_input: torch.Tensor) -> torch.Tensor:
        """Advance every fly by one internal spiking sub-step. `external_input` must be dense
        int64, shape `(n_flies, n_neurons)` -- pass a zero tensor for "no injection this step" (the
        made-up/neutral input Step 2's own "done when" criterion allows). Returns this step's spike
        tensor, same shape, values in `{0, 1}`.
        """
        expected_shape = (self.n_flies, self.connectivity.n_neurons)
        if tuple(external_input.shape) != expected_shape:
            raise ValueError(
                f"external_input has shape {tuple(external_input.shape)}, expected "
                f"{expected_shape} ({self.n_flies} fly(s) x {self.connectivity.n_neurons} neurons)"
            )
        self.state = step(self.state, self.connectivity.weight_matrix, external_input, self.params)
        self.t_step += 1
        return self.state.spikes

    def run(self, n_steps: int, external_input: torch.Tensor | None = None) -> torch.Tensor:
        """Convenience for "run N steps, collect every step's spikes" -- mainly for tests and the
        Kaggle-only full-connectome check (`scripts/run_full_connectome_check.py`), not something
        Step 4's real per-world-tick usage will call directly (that reconciles its own `dt` against
        however many internal sub-steps it needs per call -- see docs/M1_PLAN.md's Step 2 "clock
        ratio" decision -- rather than running a fixed `n_steps` up front).

        `external_input`: same shape as one step's input, reused unchanged every step, or `None`
        for all-zero (no injection) every step. Returns a stacked spike history, shape
        `(n_steps, n_flies, n_neurons)`.
        """
        if external_input is None:
            external_input = torch.zeros((self.n_flies, self.connectivity.n_neurons), dtype=torch.int64, device=self._device)
        history = []
        for _ in range(n_steps):
            history.append(self.step(external_input))
        return torch.stack(history)
