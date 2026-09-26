"""The toy brain: a small, fake stand-in for the real 166k-neuron connectome, running through the
exact same plumbing shape the real one will eventually use, so that plumbing (steering, spawning,
feeding, walls) gets built and debugged now, cheaply, rather than waiting on the real dataset.

What's real about the shape, per the plan's own design decisions for "the brain":
  - a persistent per-fly activity state, advanced by a leaky rate-based update each tick (not a
    stateless steering formula the way `Baseline` is -- that's the actual difference this project
    is built to measure, and the toy brain needs to *be* that shape, not just resemble the result)
  - sensory input driven "the simple way": a real stimulus (odour concentration, wind, nearby
    agents/walls) drives a named input channel harder the stronger it is, standing in for
    Or42b/Gr64f, Johnston's organ, and LC16 respectively -- taste (Gr64f/Gr5a) and the actual
    feeding timer are already handled by `agents.py`'s capture-radius logic and aren't repeated here
  - descending output read as turn rate (left-minus-right, the DNa02/DNa01/PFL2 shape) and forward
    speed (DNp09), including its documented "very high drive = stop" quirk -- see `_forward_speed`
  - non-holonomic locomotion: a fly turns, then walks forward relative to its own heading, not a
    free vector in an arbitrary direction the way `Baseline` moves. Heading is state this brain
    owns per agent (`Agent` itself has none -- `Population` doesn't need a heading, only a
    brain-driven agent does), the same pattern `Baseline` already uses for its own per-agent state
  - "one shared weight matrix for all flies": the recurrent/sensory/output matrices below are
    built once, from a fixed structural seed, and every fly's own personality, activity state and
    per-fly noise draws are what make one fly's behaviour differ from another's on the same brain
  - bounded activity from the start ("guard rails on activity from day one") -- the leaky update's
    own tanh already keeps it bounded, and it's clipped again on top of that, belt-and-suspenders
  - reproducible: every fly gets its own `numpy.random.Generator` seeded from
    `(scenario_seed, agent_id)`, not a shared stream, so the same scenario seed always produces
    the same fly behaviour regardless of what order ids happen to be processed in

What's NOT real, and deliberately so, until the actual connectome lands:
  - the recurrent weight matrix (`_W`) is small and randomly generated, not the real wiring --
    it contributes texture/individuality on top of a hand-designed "reflex" pathway (turn/forward
    driven directly from the sensory channels) that guarantees basic food-seeking and
    wall/crowd-avoidance competence even though nothing here has been trained on anything. The
    real brain's competence comes from evolved wiring that's been separately validated (Shiu et
    al. 2024); this stand-in earns its competence by construction instead, and says so.
  - the 20-row personality table is a placeholder (see `placeholder_personality_table`) -- "the
    actual 20-row personality table" is explicitly an M2 task in docs/PLAN.md, not built here.
    What's here is enough variation on the plan's own six dials to prove personality actually
    changes behaviour, not a tuned set of archetypes.
  - no backward locomotion (MDN, "net speed = forward minus backward" in the plan). Left out for
    v1 to keep scope tractable -- forward speed is clamped to zero rather than going negative.
    A real cut, recorded here rather than silently dropped.
  - no mate-seeking visual channel (LC10) and no dedicated "stop" neurons -- both already named as
    stretch/M2 items in the plan, not v1 gaps specific to this file.

    brain = ToyBrain(walkable_map, scenario.targets, scenario_seed=scenario.meta.seed)
    vel = brain.desired_velocities(ids, positions, radii, max_speed_mps, personalities, odor_field, dt)
    brain.forget(pop.leave())
"""

from __future__ import annotations

import numpy as np

from ..world.geodesic import gradient_direction
from ..world.map import WalkableMap
from .repulsion import agent_repulsion, wall_repulsion
from .targeting import TargetAssignment

PERSONALITY_DIALS = (
    "sugar_sensitivity",
    "mate_sensitivity",
    "timidity",
    "boldness",
    "baseline_speed",
    "noise",
)


def placeholder_personality_table(n: int = 20, seed: int = 20260000) -> dict[str, np.ndarray]:
    """A stand-in 20-row personality table -- NOT "the actual 20-row personality table" the plan
    names as an M2 task (hand-picked archetypes like hungry-and-bold, slow-and-steady, and the
    blends between them). This is just enough reproducible variation on the same six dials to
    prove the plumbing actually responds to personality, not a tuned design. Swap this for the
    real table wholesale when that M2 work happens; nothing else here should need to change, since
    consumers only ever look values up by dial name."""
    rng = np.random.default_rng(seed)
    return {
        "sugar_sensitivity": rng.uniform(0.6, 1.4, n),
        "mate_sensitivity": rng.uniform(0.6, 1.4, n),
        "timidity": rng.uniform(0.0, 1.0, n),
        "boldness": rng.uniform(0.0, 1.0, n),
        "baseline_speed": rng.uniform(0.75, 1.25, n),
        "noise": rng.uniform(0.0, 1.0, n),
    }


_N_SENSE = 6  # food_ahead, food_left, wind_ahead, wind_left, avoid_ahead, avoid_left
_HIDDEN = 16
_ACTIVITY_MAX = 5.0  # guard rail on top of tanh's own bound
_TAU_S = 0.3  # leaky integration time constant, seconds
_TURN_GAIN = 1.0  # rad/s per unit of net turn signal
_TURN_MAX = 3.0  # guard rail on the net turn signal itself, before `_TURN_GAIN` is applied --
# heading is integrated (this tick's turn compounds onto every previous tick's), unlike
# `Baseline`'s steering, which is recomputed fresh each tick and can't accumulate. Found by
# actually running several agents close together, not just one in the open: a sustained
# avoidance reading with no guard rail here integrated into agents spinning at 400+ degrees a
# second instead of steering, because nothing was pulling the turn signal itself back into a
# realistic range the way `_ACTIVITY_MAX` already does for the hidden state.
_NOISE_SCALE = 1.5  # rad/s per unit of the noise dial, at a full-strength random draw
_HIDDEN_TURN_WEIGHT = 0.4  # how much the random recurrent net can nudge turning...
_HIDDEN_FWD_WEIGHT = 0.3  # ...and forward drive, on top of the designed reflex pathway
_K_FOOD_TURN = 2.0
_K_AVOID_TURN = 3.0
_K_FOOD_FWD = 0.6
_K_AVOID_FWD = 1.0
_K_BASE_FWD = 0.7


class ToyBrain:
    """A small rate-based network standing in for one fly's brain -- see the module docstring for
    exactly what's real about its shape and what's a placeholder."""

    def __init__(
        self,
        walkable_map: WalkableMap,
        targets,
        scenario_seed: int = 0,
        personality_table: dict[str, np.ndarray] | None = None,
        structural_seed: int = 20260000,
    ):
        self.map = walkable_map
        self.targets = list(targets)  # scenario.targets-like: each needs .position and .kind
        if not self.targets:
            raise ValueError("ToyBrain needs at least one target to steer toward")
        self._targets = TargetAssignment(walkable_map, [t.position for t in self.targets])
        self.scenario_seed = int(scenario_seed)
        self.personality_table = (
            personality_table if personality_table is not None else placeholder_personality_table()
        )

        rng = np.random.default_rng(structural_seed)
        self._W = rng.normal(0.0, 1.0 / np.sqrt(_HIDDEN), size=(_HIDDEN, _HIDDEN))
        self._Win = rng.normal(0.0, 0.5, size=(_HIDDEN, _N_SENSE))
        self._Wout = rng.normal(0.0, 0.3, size=(2, _HIDDEN))  # -> [turn_hidden, forward_hidden]

        self._activity: dict[int, np.ndarray] = {}
        self._heading: dict[int, float] = {}
        self._rng: dict[int, np.random.Generator] = {}
        # The plan's whole "replay debugger for agents" design rests on being able to recompute
        # any one fly's brain later from just its recorded sensory inputs, without re-running the
        # rest of the crowd. `last_sense` is what a recorder (see `flybrainflow/recorder.py`)
        # actually captures each tick -- the exact 6-channel input this fly's brain saw, keyed by
        # id. It's overwritten every tick, not accumulated; a recorder must read it right after
        # calling `desired_velocities`, before the next tick overwrites it.
        self.last_sense: dict[int, np.ndarray] = {}

    def assigned_targets(self) -> dict[int, int]:
        """The live id -> committed-target-index mapping -- same contract as
        `Baseline.assigned_targets`, so `Sim.tick` can ask either brain the same way rather than
        reaching into `self._targets` directly."""
        return self._targets.assigned

    def forget(self, ids) -> None:
        """Release every piece of per-agent state for ids that are gone for good -- same idea,
        same reason, as `Baseline.forget`: without this, all four dicts below (plus the target
        assignment cache) grow for as long as the process runs, not just for as long as an agent
        is actually alive. Forgetting an id that's still walking is harmless; it's just
        re-initialised the next time it's seen, as if newly spawned."""
        for i in ids:
            self._activity.pop(i, None)
            self._heading.pop(i, None)
            self._rng.pop(i, None)
            self.last_sense.pop(i, None)
        self._targets.forget(ids)

    def move_target(self, index: int, new_xy) -> None:
        """Relocate target `index` to `new_xy` -- see `TargetAssignment.move_target` for the full
        rationale (already-assigned flies stay committed to `index` and just get a correct field
        for its new position; nothing needs reassigning). Unlike `Baseline`, `ToyBrain` has no
        `self.target_positions` alias into `self._targets` -- callers that need the live positions
        read `self._targets.target_positions` (or `self.targets[i].position`, which this call does
        *not* update: `self.targets` is the original `scenario.targets`-like objects used for
        `.kind` lookups in `_sense`, not a position source)."""
        self._targets.move_target(index, new_xy)

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
        """One steering vector per id in `ids`, in the same order -- same call shape as
        `Baseline.desired_velocities` (positional `ids, positions, radii, max_speed_mps,
        preferred_targets`, then keyword-only `personalities, odor_field, dt`), so `Sim._steer`
        calls both brains identically. `Baseline` accepts the three keyword-only arguments and
        ignores them; `ToyBrain` actually needs all three: `personalities` (this fly's row into
        `self.personality_table`, e.g. `Agent.personality`), `odor_field` (a `world.fields.OdorField`
        built for the same map, so sensing is wall-aware/wind-bent exactly like a real fly's would
        be), and `dt` (the leaky activity update needs real elapsed time, unlike `Baseline`'s
        stateless formula) -- kept required (no default) here since a toy or real connectome brain
        genuinely cannot function without them. `preferred_targets` (id -> target index) overrides
        nearest-distance assignment for an id's first assignment only -- see
        `TargetAssignment.assign`."""
        positions = np.asarray(positions, float).reshape(-1, 2)
        n = len(ids)
        if n == 0:
            return np.zeros((0, 2))
        radii = np.broadcast_to(np.asarray(radii, float), (n,))
        max_speed = np.broadcast_to(np.asarray(max_speed_mps, float), (n,))
        personalities = np.broadcast_to(np.asarray(personalities, int), (n,))

        target_idx = self._targets.assign(ids, positions, preferred_targets)
        self._ensure_state(ids, positions, target_idx)
        heading = np.array([self._heading[i] for i in ids])
        heading_vec = np.c_[np.cos(heading), np.sin(heading)]

        sense, food_direction, food_drive = self._sense(ids, positions, radii, target_idx, heading_vec, odor_field)
        for k, i in enumerate(ids):
            self.last_sense[i] = sense[k]
        activity = np.stack([self._activity[i] for i in ids])
        drive = sense @ self._Win.T
        recur = activity @ self._W.T
        activity = activity + (dt / _TAU_S) * (-activity + np.tanh(recur + drive))
        activity = np.clip(activity, -_ACTIVITY_MAX, _ACTIVITY_MAX)
        hidden_out = activity @ self._Wout.T  # (n, 2) -> [turn_hidden, forward_hidden]

        table = self.personality_table
        sugar_sens = table["sugar_sensitivity"][personalities]
        mate_sens = table["mate_sensitivity"][personalities]
        timidity = table["timidity"][personalities]
        boldness = table["boldness"][personalities]
        baseline_speed = table["baseline_speed"][personalities]
        noise = table["noise"][personalities]
        kinds = np.array([t.kind for t in self.targets])[target_idx]
        food_sens = np.where(kinds == "sugar", sugar_sens, mate_sens)
        # `food_ahead`/`food_left` in `sense` are already scaled by concentration; personality
        # sensitivity is a further per-fly multiplier on top of that shared stimulus.
        food_ahead, food_left = sense[:, 0] * food_sens, sense[:, 1] * food_sens
        avoid_ahead, avoid_left = sense[:, 4], sense[:, 5]
        avoid_gain = np.clip(1.0 + timidity - boldness, 0.2, 2.0)

        # The hand-designed reflex's turn term used to be `_K_FOOD_TURN * food_left`, the same
        # bilateral (left-vs-right) cross-product signal fed to the recurrent net. That signal is
        # mathematically zero both when a fly is facing its target dead-on (0 degrees off) AND when
        # it's facing directly away from it (180 degrees off) -- found by actually watching several
        # agents converge on one target rather than testing one fly alone: once avoidance rotated a
        # fly past roughly 90 degrees off course, there was nothing pulling it back, and it would
        # just cruise away from its target at baseline speed forever. A bilateral signal can tell
        # "should I nudge left or right" near zero error, but it can't tell "which way is even
        # forward from here" once the error gets large -- there's no substitute for closing the
        # loop on the actual heading error. So the *designed* reflex now steers on a proper
        # heading-error (P-controller) term instead; the bilateral `food_left` stays in `sense` as
        # a secondary, smaller-weighted (`_HIDDEN_TURN_WEIGHT`) input to the recurrent net, which is
        # free to use it for finer-grained texture on top of the guaranteed-competence reflex below.
        desired_heading = np.arctan2(food_direction[:, 1], food_direction[:, 0])
        heading_error = np.mod(desired_heading - heading + np.pi, 2 * np.pi) - np.pi
        turn_food = _K_FOOD_TURN * heading_error * food_drive * food_sens

        turn_noise = np.array([self._rng[i].normal() for i in ids]) * noise * _NOISE_SCALE
        turn = (
            turn_food
            - _K_AVOID_TURN * avoid_gain * avoid_left
            + _HIDDEN_TURN_WEIGHT * hidden_out[:, 0]
            + turn_noise
        )
        turn = np.clip(turn, -_TURN_MAX, _TURN_MAX)
        forward_drive = (
            _K_BASE_FWD * baseline_speed
            + _K_FOOD_FWD * np.clip(food_ahead, 0.0, None)
            - _K_AVOID_FWD * avoid_gain * np.clip(avoid_ahead, 0.0, None)
            + _HIDDEN_FWD_WEIGHT * hidden_out[:, 1]
        )
        forward_speed = max_speed * _forward_speed_curve(np.clip(forward_drive, 0.0, None))

        new_heading = heading + turn * _TURN_GAIN * dt
        new_heading = np.mod(new_heading + np.pi, 2 * np.pi) - np.pi
        velocity = forward_speed[:, None] * np.c_[np.cos(new_heading), np.sin(new_heading)]

        for k, i in enumerate(ids):
            self._activity[i] = activity[k]
            self._heading[i] = float(new_heading[k])
        return velocity

    # -- internals ------------------------------------------------------

    def _ensure_state(self, ids, positions: np.ndarray, target_idx: np.ndarray) -> None:
        new_heading_rows = []
        for k, (i, ti) in enumerate(zip(ids, target_idx)):
            if i not in self._rng:
                self._rng[i] = np.random.default_rng((self.scenario_seed, int(i)))
            if i not in self._activity:
                self._activity[i] = np.zeros(_HIDDEN)
            if i not in self._heading:
                new_heading_rows.append(k)
        if new_heading_rows:
            # Face toward the assigned target at birth rather than an arbitrary direction -- a
            # fly that spawns facing a wall behind it has nothing useful to steer by yet.
            for k, ti in zip(new_heading_rows, target_idx[new_heading_rows]):
                field = self._targets.field_for(int(ti))
                direction = -gradient_direction(self.map, field, positions[k : k + 1])[0]
                heading = 0.0 if np.allclose(direction, 0.0) else float(np.arctan2(direction[1], direction[0]))
                self._heading[ids[k]] = heading

    def _sense(self, ids, positions, radii, target_idx, heading_vec, odor_field) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns `(sense, food_direction, food_drive)`. `sense` is the bilateral (ahead/left)
        array fed to the recurrent net -- fine as a *secondary* input, but see below for why it's
        not enough on its own. `food_direction` (n, 2) and `food_drive` (n,) are the world-frame
        unit vector toward stronger smell and how strong that smell is, kept around so the caller
        can build a proper heading-error steering signal from them."""
        n = len(ids)
        sense = np.zeros((n, _N_SENSE))
        food_direction = np.zeros((n, 2))
        food_drive = np.zeros(n)
        for ti in np.unique(target_idx):
            rows = target_idx == ti
            p = positions[rows]
            # Sampling by `kind` alone would sum *every* target of that kind into one combined
            # plume -- fine for "how strong does sugar smell overall", wrong for "which way should
            # I walk" once two same-kind targets exist, since a fly committed to a far target would
            # still feel (and get pulled off course by) a near target's smell just because they're
            # the same kind. See `OdorField.gradient`'s docstring for the full mechanism.
            # `odor_field.sources` is built 1:1, in order, from the same `scenario.targets` this
            # brain's own `self.targets` came from (see `world.fields.from_scenario` and
            # `Sim.__init__`), so index `ti` picks out exactly the one plume this fly has committed
            # to.
            source = odor_field.sources[int(ti)] if int(ti) < len(odor_field.sources) else None
            kind = self.targets[int(ti)].kind
            conc = odor_field.sample(p, kind=kind, source=source)
            grad = odor_field.gradient(p, kind=kind, source=source)
            gnorm = np.linalg.norm(grad, axis=1)
            zero = gnorm < 1e-9
            direction = np.zeros_like(grad)
            direction[~zero] = grad[~zero] / gnorm[~zero, None]
            drive = np.tanh(conc)
            hv = heading_vec[rows]
            sense[rows, 0] = _dot(hv, direction) * drive
            sense[rows, 1] = _cross_z(hv, direction) * drive
            food_direction[rows] = direction
            food_drive[rows] = drive

        local_wind = odor_field.local_wind(positions)
        wspeed = np.linalg.norm(local_wind, axis=1)
        zero = wspeed < 1e-9
        wind_unit = np.zeros_like(local_wind)
        wind_unit[~zero] = local_wind[~zero] / wspeed[~zero, None]
        wind_drive = np.tanh(wspeed)
        sense[:, 2] = _dot(heading_vec, wind_unit) * wind_drive
        sense[:, 3] = _cross_z(heading_vec, wind_unit) * wind_drive

        avoid_force = agent_repulsion(positions, radii) + wall_repulsion(self.map, positions)
        amag = np.linalg.norm(avoid_force, axis=1)
        zero = amag < 1e-9
        avoid_unit = np.zeros_like(avoid_force)
        avoid_unit[~zero] = avoid_force[~zero] / amag[~zero, None]
        avoid_drive = np.tanh(amag)
        sense[:, 4] = _dot(heading_vec, avoid_unit) * avoid_drive
        sense[:, 5] = _cross_z(heading_vec, avoid_unit) * avoid_drive
        return sense, food_direction, food_drive


def _forward_speed_curve(drive: np.ndarray) -> np.ndarray:
    """The plan's documented DNp09 quirk: very high drive acts as a brake, not a ceiling. Ramps up
    roughly linearly for drive in [0, 1], then decays back down past it, instead of saturating
    flat -- so an agent that's simultaneously being told "go" very hard by several channels at
    once slows back down rather than just capping at top speed."""
    return drive * np.exp(-np.clip(drive - 1.0, 0.0, None))


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _cross_z(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
