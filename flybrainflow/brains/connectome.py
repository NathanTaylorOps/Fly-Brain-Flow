"""`ConnectomeBrain` -- Step 4's real brain, wiring the actual MaleCNS spiking connectome
(`flybrainflow.spiking`) into the same `desired_velocities`/`assigned_targets`/`forget`/
`move_target`/`last_sense` interface `ToyBrain` (see `toy.py`) and `Baseline` already implement, so
`Sim._steer` can drive this brain exactly the way it drives either of those, unchanged.

**THIS IS A SKELETON, NOT STEP 4.** Written 2026-09-27 night, while Nathan was asleep, specifically
to sketch the SHAPE of the real work without pretending any of it is done -- see
`flybrainflow/spiking/__init__.py`'s own docstring and docs/M1_PLAN.md's Step 4 section for the full
reasoning already recorded earlier tonight: Step 4 is deliberately gated behind Step 3's calibration
gate actually PASSING, because Step 4's own key parameters (sensory-injection scaling, the clock
ratio between one world tick and however many internal spiking sub-steps it needs) are supposed to
come FROM that result, not be guessed in advance. That gate hasn't run against real data yet (see
docs/M1_PLAN.md's Step 3 pre-work section for the two open gaps -- MN9/sugar-GRN naming, FlyWire
access -- both researched tonight but neither confirmed live).

So every method below either raises `NotImplementedError` naming exactly what real, calibrated input
it's missing, or is left undefined entirely. Every constructor argument that would need a real
number from Step 3 is REQUIRED, with no default -- there is no "reasonable guess" for a
sensory-injection scale or a clock ratio that this project's own stated discipline (`docs/PLAN.md`'s
"one variable at a time") would accept as a placeholder. This file exists so that when Step 3 does
pass, the actual Step 4 work is "fill in these specific numbers and bodies of these specific
methods" rather than "start from a blank file and re-derive the interface shape `ToyBrain` already
worked out." Nothing here has been run, tested, or checked against real data -- there is no real
data flowing through any of it yet to test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spiking import SpikingSimulator


@dataclass(frozen=True)
class SensoryWiring:
    """Which real MaleCNS body-ids each of the toy brain's sensory channels (see `toy.py`'s own
    `_N_SENSE` comment: food_ahead/food_left, wind_ahead/wind_left, avoid_ahead/avoid_left) actually
    inject current into, and how strongly. NOT populated with real values anywhere yet -- every
    field here needs a real body-id array from `flybrainflow.spiking.body_ids_for_type` run against
    the real, live annotation table, plus an injection-strength constant Step 3's calibration run is
    supposed to inform (see this module's own docstring). Constructing one of these with placeholder
    arrays would be indistinguishable, later, from constructing one with real, checked values --
    exactly the silent-placeholder failure mode this project's own discipline tries to avoid
    elsewhere (e.g. `data_config.KNOWN_GAPS` being loud about what's confirmed absent, not just
    silently working with fewer neurons). So this dataclass has no default values and no
    `from_defaults()`/`placeholder()` constructor -- whoever builds one must supply real arrays,
    sourced from a real, live lookup, on purpose.
    """

    food_odor_body_ids: np.ndarray  # Or42b-equivalent -- CONFIRMED ABSENT from MaleCNS by name (data_config.KNOWN_GAPS); needs the Step 4 sensory-model decision that section's own comment defers
    food_taste_body_ids: np.ndarray  # Gr64f/Gr5a-equivalent -- see the LB3b/LB3c research candidate in data_config.py; unconfirmed live
    wind_body_ids: np.ndarray  # Johnston's organ (data_config.JOHNSTONS_ORGAN_PREFIX, "JO-.*") -- present and typed, unlike the two above, but not yet wired to a specific injection scheme
    avoid_visual_body_ids: np.ndarray  # LC16-equivalent, looming/avoidance -- present per data_config.REQUIRED_NEURON_TYPES, not yet wired
    injection_strength: float  # scales sensory drive -> raw current injected per tick; NO sensible default exists before Step 3's calibration run gives a feel for real firing-rate scales


@dataclass(frozen=True)
class MotorReadout:
    """Which real MaleCNS body-ids drive turning and forward speed, mirroring `toy.py`'s own
    DNa02/DNa01/PFL2 (turn) and DNp09 (forward) reflex pathway -- same "no placeholder values"
    reasoning as `SensoryWiring` above."""

    turn_body_ids: np.ndarray  # DNa02/DNa01/PFL2-equivalent
    forward_body_ids: np.ndarray  # DNp09-equivalent
    readout_scale: float  # spike rate -> turn-rate/forward-speed units; no default, same reasoning as SensoryWiring.injection_strength


class ConnectomeBrain:
    """Step 4's real brain -- SKELETON ONLY, see module docstring. Interface deliberately mirrors
    `ToyBrain` exactly (`desired_velocities`, `assigned_targets`, `forget`, `move_target`,
    `last_sense`) so `Sim._steer` needs no brain-specific branching to drive either one.
    """

    def __init__(
        self,
        simulator: SpikingSimulator,
        sensory_wiring: SensoryWiring,
        motor_readout: MotorReadout,
        *,
        brain_dt: float,
        sub_steps_per_tick: int,
    ) -> None:
        """`simulator`: an already-built `SpikingSimulator` over the real, signed, calibrated
        MaleCNS connectivity (sign resolution via `flybrainflow.spiking.resolve_signs`/
        `apply_signs` -- see `neurotransmitters.py`) -- this class does not build one itself,
        the same separation-of-concerns `ToyBrain` keeps from `WalkableMap`/`OdorField`.

        `brain_dt`: seconds of simulated brain time per internal spiking sub-step. `sub_steps_per_tick`:
        how many of those sub-steps run per one call to `desired_velocities` (one world tick). Both
        REQUIRED, no default -- this is exactly the "clock ratio... an empirical call... once Step
        3's calibration run gives a feel for it" decision docs/M1_PLAN.md's own Step 4 text names as
        still open. Passing a guessed value here would silently satisfy this constructor while
        actually reintroducing the exact guessed-in-advance parameter Step 3 exists to avoid --
        so this constructor does not stop you from passing a bad guess (it can't know a good value
        from a bad one), but the absence of a default at least stops a caller from getting one for
        free without deciding it explicitly.
        """
        self.simulator = simulator
        self.sensory_wiring = sensory_wiring
        self.motor_readout = motor_readout
        self.brain_dt = float(brain_dt)
        self.sub_steps_per_tick = int(sub_steps_per_tick)
        if self.sub_steps_per_tick < 1:
            raise ValueError(f"sub_steps_per_tick must be >= 1, got {self.sub_steps_per_tick}")

        self._heading: dict[int, float] = {}
        # `last_sense`/`_SENSE_WIDTH` -- see docs/M1_PLAN.md's Step 2 status section and Step 3's
        # pre-work list: Step 2's own text promised this becomes brain-reported, not hardcoded, and
        # that was never actually done. Left as an explicit TODO here rather than picking an
        # arbitrary width, for the same reason `SensoryWiring` has no placeholder constructor --
        # a wrong-but-silently-working width is worse than an honest gap.
        self.last_sense: dict[int, np.ndarray] = {}

    def assigned_targets(self) -> dict[int, int]:
        raise NotImplementedError(
            "ConnectomeBrain is a Step 4 skeleton, not a working brain -- target assignment needs "
            "the same TargetAssignment machinery ToyBrain uses (see toy.py), not yet wired here "
            "because there's no real sensory/motor mapping to steer toward a target with yet. See "
            "flybrainflow/spiking/__init__.py's own docstring for why Step 4 is gated behind Step "
            "3's calibration gate passing on real data."
        )

    def forget(self, ids) -> None:
        raise NotImplementedError(
            "ConnectomeBrain is a Step 4 skeleton -- no per-agent state exists yet to forget "
            "(self._heading/self.last_sense are declared but never populated by a working "
            "desired_velocities). See this module's own docstring."
        )

    def move_target(self, index: int, new_xy) -> None:
        raise NotImplementedError(
            "ConnectomeBrain is a Step 4 skeleton -- no target-assignment machinery is wired yet. "
            "See this module's own docstring."
        )

    def desired_velocities(
        self,
        ids,
        positions,
        radii,
        max_speed_mps,
        preferred_targets: dict | None = None,
        *,
        personalities,
        odor_field,
        dt: float,
    ) -> np.ndarray:
        """Same call shape as `ToyBrain.desired_velocities`/`Baseline.desired_velocities` on
        purpose, so `Sim._steer` can call any of the three brains identically -- but this one
        raises, always, because the real work (inject `odor_field` samples into
        `sensory_wiring`'s body-ids as spiking-simulator input, run `sub_steps_per_tick` internal
        steps, read `motor_readout`'s body-ids' spike rates back out as turn/forward drive, same
        general shape as `ToyBrain`'s reflex pathway but driven by real wiring instead of a
        hand-designed formula) genuinely cannot be written responsibly yet: `sensory_wiring`'s and
        `motor_readout`'s body-id arrays would have to come from real, live-confirmed cell-type
        lookups this project doesn't have yet (see docs/M1_PLAN.md's Step 3 pre-work section), and
        `injection_strength`/`readout_scale`/`brain_dt`/`sub_steps_per_tick` would have to come from
        Step 3's own calibration result, which hasn't run against real data yet either. Filling this
        in now would mean picking numbers with nothing to check them against -- exactly what this
        project's "one variable at a time" discipline (docs/PLAN.md) argues against.
        """
        raise NotImplementedError(
            "ConnectomeBrain.desired_velocities is not implemented -- this class is a Step 4 "
            "interface skeleton only. Step 4 is gated behind Step 3's calibration gate passing on "
            "real MaleCNS data (see docs/M1_PLAN.md's Step 3/4 sections); implement this method "
            "once that gate has passed and its result gives real values for SensoryWiring's "
            "injection_strength, MotorReadout's readout_scale, and this class's brain_dt/"
            "sub_steps_per_tick, not before."
        )
