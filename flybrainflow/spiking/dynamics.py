"""The leaky-integrate-and-fire (LIF) update rule -- one internal spiking sub-step for the whole
population at once, for one or more flies in a batch. Knows nothing about neuron identity, body
IDs, or where the connectome came from -- just integer arrays shaped `(n_flies, n_neurons)`.

Everything here is deliberately integer arithmetic, not float, per docs/M1_PLAN.md's own Step 2:
"Build the update rule around integer (fixed-point) spike accumulation... Floating-point atomic-add
spike accumulation on GPU is not bit-reproducible run to run... batched, parallel reductions can
land in a different order each time." Integer addition is associative and exact -- unlike float
addition, the same set of integers summed in a different order always gives the identical result,
which is the actual property that makes a GPU-batched run reproducible, not merely
`torch.use_deterministic_algorithms(True)` on its own (that flag picks a deterministic *algorithm*
for a given op; it doesn't make floating-point summation associative).

One real wrinkle: `torch.sparse.mm` needs a floating dtype, so the sparse weight matrix @ spike
vector step can't be done in int64 directly. The fix used here is the standard "float64 as an exact
integer accumulator" trick -- float64 represents every integer up to 2**53 exactly, and summing
exactly-representable integers whose running total never exceeds 2**53 is itself exact (no
rounding occurs, so no rounding-order sensitivity either). Real numbers here are nowhere close to
that ceiling (worst case: every one of ~165k neurons firing at once into one neuron at the real
data's observed max weight of 2591 sums to ~4.3e8, twenty million times smaller than 2**53), so the
matmul is done in float64 and cast back to int64 exactly afterward -- not an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# float64 exactly represents every integer up to this; see module docstring. Real MaleCNS-scale
# sums are ~2e16 times smaller than this, so this is a headroom check, not a real limit anyone
# should ever get close to.
_EXACT_INT_CEILING = 2**53


@dataclass(frozen=True)
class LIFParams:
    """Tuned during Step 3's calibration gate -- these defaults are placeholders, not calibrated
    values, and Step 2's own "done when" only needs *some* stable, non-blowing-up parameter set to
    prove the scaffold runs; picking the biologically-right numbers is explicitly out of scope
    here.
    """

    threshold: int = 100  # membrane potential at/above which a neuron fires
    leak_numerator: int = 9  # decay applied each step as (V * leak_numerator) // leak_denominator
    leak_denominator: int = 10  # -- 9/10 means "keep 90% of the previous potential" per step
    refractory_ticks: int = 2  # ticks a neuron can't fire again after it just fired
    activity_min: int = -1000  # guard rail -- see docs/M1_PLAN.md's Step 2 "guard rails from day one"
    activity_max: int = 1000  # guard rail -- must be >= threshold or a neuron could never fire
    reset_value: int = 0  # membrane potential immediately after a spike

    def __post_init__(self) -> None:
        if self.activity_max < self.threshold:
            raise ValueError(
                f"activity_max ({self.activity_max}) is below threshold ({self.threshold}) -- "
                "a neuron could never reach threshold to fire; this is almost certainly a "
                "misconfiguration, not an intentional guard rail"
            )
        if self.activity_min > self.reset_value:
            raise ValueError(
                f"activity_min ({self.activity_min}) is above reset_value ({self.reset_value}) -- "
                "a neuron would leave its own guard rail immediately after every spike"
            )
        if not (0 <= self.leak_numerator <= self.leak_denominator):
            raise ValueError(
                f"leak_numerator/leak_denominator ({self.leak_numerator}/{self.leak_denominator}) "
                "must satisfy 0 <= leak_numerator <= leak_denominator -- this is a decay factor, "
                "not a growth factor, and a negative ratio makes no physical sense"
            )


@dataclass
class LIFState:
    """Everything that carries from one tick to the next, for `n_flies` flies over `n_neurons`
    neurons each. All three tensors share shape `(n_flies, n_neurons)`.
    """

    membrane: torch.Tensor  # int64 -- current membrane potential
    refractory_remaining: torch.Tensor  # int64 -- ticks left before this neuron can fire again
    spikes: torch.Tensor  # int64, values in {0, 1} -- whether this neuron fired *last* tick
    # (last tick's spikes, not this tick's -- `step()` reads this to compute this tick's synaptic
    # input, then overwrites it with this tick's own spikes before returning)

    @classmethod
    def zeros(cls, n_flies: int, n_neurons: int, params: LIFParams, device: str | torch.device | None = None) -> "LIFState":
        shape = (n_flies, n_neurons)
        return cls(
            membrane=torch.full(shape, params.reset_value, dtype=torch.int64, device=device),
            refractory_remaining=torch.zeros(shape, dtype=torch.int64, device=device),
            spikes=torch.zeros(shape, dtype=torch.int64, device=device),
        )


def _synaptic_input(weight_matrix: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
    """`weight_matrix` is sparse int64, shape (n, n), post-major (see connectivity.py). `spikes` is
    dense int64, shape (n_flies, n). Returns dense int64, shape (n_flies, n): each fly's total
    incoming synaptic input per neuron this tick. See module docstring for why this goes through
    float64 rather than doing the matmul in int64 directly.
    """
    n_flies, n = spikes.shape
    max_possible_sum = int(weight_matrix.values().abs().sum().item()) if weight_matrix.values().numel() else 0
    if max_possible_sum >= _EXACT_INT_CEILING:
        raise OverflowError(
            f"sum of |weights| ({max_possible_sum}) approaches float64's exact-integer ceiling "
            f"({_EXACT_INT_CEILING}) -- the float64-as-exact-integer-accumulator trick this "
            "function relies on (see dynamics.py's module docstring) is no longer safe at this "
            "scale; this would need a real fixed-point/bigint accumulation strategy instead"
        )
    weight_f64 = weight_matrix.to(torch.float64)
    spikes_f64 = spikes.to(torch.float64)
    # weight_matrix @ spikes.T gives (n, n_flies); transpose back to the (n_flies, n) shape
    # everything else in this module uses.
    result_f64 = torch.sparse.mm(weight_f64, spikes_f64.T).T
    return torch.round(result_f64).to(torch.int64)


def step(state: LIFState, weight_matrix: torch.Tensor, external_input: torch.Tensor, params: LIFParams) -> LIFState:
    """One internal spiking sub-step for the whole population, all flies at once.

    `external_input`: dense int64, shape (n_flies, n_neurons) -- this tick's externally-injected
    current (sensory input, Step 4's job to populate meaningfully; Step 2's own tests use made-up
    values, per its "done when" criterion). Add zeros for neurons nothing is injecting into.
    """
    synaptic = _synaptic_input(weight_matrix, state.spikes)

    decayed = torch.div(state.membrane * params.leak_numerator, params.leak_denominator, rounding_mode="floor")
    in_refractory = state.refractory_remaining > 0

    driven = decayed + synaptic + external_input
    clipped = torch.clamp(driven, params.activity_min, params.activity_max)
    # A refractory neuron doesn't integrate new input at all -- its membrane potential is simply
    # frozen (not clipped/decayed either) until refractory_remaining reaches zero. This is a
    # deliberate modeling choice (the simplest correct LIF refractory behavior), not an oversight.
    new_membrane = torch.where(in_refractory, state.membrane, clipped)

    will_spike = (~in_refractory) & (new_membrane >= params.threshold)
    new_membrane = torch.where(will_spike, torch.full_like(new_membrane, params.reset_value), new_membrane)

    new_refractory = torch.where(
        will_spike,
        torch.full_like(state.refractory_remaining, params.refractory_ticks),
        torch.clamp(state.refractory_remaining - 1, min=0),
    )

    return LIFState(
        membrane=new_membrane,
        refractory_remaining=new_refractory,
        spikes=will_spike.to(torch.int64),
    )
