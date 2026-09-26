"""Agent lifecycle: spawn, feed, leave.

This doesn't move anyone (that's `world.physics`) and doesn't decide where an agent wants to go
(that's the brain, real or toy, still to come). It's the bookkeeping the scenario file already
promises and nothing acts on yet: entry points and their rate, a population cap, target slots and
feeding time, and what happens once a fly actually reaches a target.

    pop = Population.from_scenario(scenario)
    # each tick, in this order:
    pop.spawn(dt)                                  # new agents at open sources, up to the cap
    pop.set_positions(ids, new_positions)          # after physics.step() moves the walking ones
    pop.feed(dt, capture_radius=0.5)               # start/continue feeding for anyone close enough
    left = pop.leave()                             # remove anyone who finished, freeing their slot

Only for `boundary_mode = "open"` scenarios (an entry-and-exit venue). A `"closed"` scenario (the
ring-road test) starts with its whole fixed population already in place and nobody ever spawns or
leaves — that's a different kind of setup, tied to the ring's own geometry, and is explicitly an
M2 task (see docs/PLAN.md's milestone list), not built here.

**Two cohorts sharing one venue.** The scenario schema's `[baseline] enabled = true` and the plan's
own "a standard crowd model running in the same space" both mean two separate crowds -- fly-brained
and baseline -- walking the same venue at the same time, not one homogeneous pool. A single
`Population` still only ever drives one cohort (it doesn't know or care which brain reads its
agents), but a target's feeding capacity and an agent's id are both venue-wide facts, not something
each cohort gets to track for itself: two `Population`s that each thought they owned a target's 6
slots would double-book it to 12, and two `Population`s that each counted ids from zero would hand
out duplicate ids the moment anything (a brain's per-agent state, a combined physics call) tried to
index both cohorts together. `build_cohorts()` wires that sharing up correctly; use it instead of
constructing two plain `Population`s by hand whenever more than one cohort is in play:

    cohorts = build_cohorts(scenario)              # {"brain": Population, "baseline": Population}
    for pop in cohorts.values():
        pop.spawn(dt)
    ids = [i for pop in cohorts.values() for i in pop.walking_ids()]
    # ... one shared physics_step() call over every id from every cohort, so they actually see and
    # avoid each other, not two simulations layered on top of one another that happen to share a
    # picture ...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_CAPTURE_RADIUS_M = 0.5


@dataclass
class Agent:
    id: int
    position: np.ndarray
    personality: int
    brain: str = "brain"  # which cohort this agent belongs to -- "brain" (fly-brained, whichever
    # brain implementation is currently active: `ToyBrain` today, the real connectome later) or
    # "baseline", when `build_cohorts()` set up more than one. A lone `Population` built directly
    # (most existing tests, and any single-cohort scenario) leaves every agent tagged "brain" --
    # it's a label for the eventual sim loop to route on, not something `Population` itself reads.
    status: str = "walking"  # "walking" or "feeding"
    target_index: int | None = None
    feed_remaining_s: float = 0.0
    # Which [[targets]] index this fly's spawn source pinned it to (`scenario.spawn.source_targets`),
    # or None for the usual nearest-target pick every brain otherwise makes for itself. Set once at
    # spawn and never changed -- see `SpawnConfig.source_targets`'s own docstring for why a plain
    # "nearest" pick silently breaks a two-stream/crossing scenario.
    preferred_target: int | None = None


class TargetSlots:
    """Feeding-slot occupancy for a scenario's targets. A target's capacity is a physical fact
    about the venue -- it has N slots, full stop -- not something each cohort gets to track for
    itself. One `Population` walking a venue alone can own one of these outright; two `Population`s
    sharing a venue (see `build_cohorts`) must share the same instance, or they'll each think they
    have the target's full capacity to themselves and double-book it."""

    def __init__(self, targets) -> None:
        self.capacity = [t.slots for t in targets]
        self.used = [0] * len(targets)

    def try_occupy(self, target_index: int) -> bool:
        """Claim a slot at `target_index` if one's free. Returns whether it succeeded."""
        if self.used[target_index] >= self.capacity[target_index]:
            return False
        self.used[target_index] += 1
        return True

    def release(self, target_index: int) -> None:
        self.used[target_index] -= 1


class IdAllocator:
    """A shared, monotonically-increasing agent-id counter. One `Population` numbering its own
    agents from zero is fine alone; two `Population`s sharing a venue must share one of these
    instead, or "agent 3" would mean two different flies depending which cohort you asked."""

    def __init__(self, start: int = 0) -> None:
        self._next = start

    def next(self) -> int:
        i = self._next
        self._next += 1
        return i


class Population:
    """Tracks every live agent in one cohort of an open-boundary scenario: who's walking, who's
    feeding where, and how many free slots each target has left. Pass `slots`/`ids` (both shared,
    from `build_cohorts` or built by hand) whenever more than one `Population` walks the same
    venue; left as `None`, this `Population` gets its own of each, correct for the common
    single-cohort case."""

    def __init__(
        self,
        scenario,
        rng: np.random.Generator | None = None,
        brain: str = "brain",
        slots: TargetSlots | None = None,
        ids: IdAllocator | None = None,
    ):
        if scenario.meta.boundary_mode != "open":
            raise ValueError(
                f"Population is for open-boundary scenarios; '{scenario.meta.name}' is "
                f"boundary_mode={scenario.meta.boundary_mode!r} (closed-mode population setup "
                "is an M2 task, tied to the venue's own geometry, not built here)"
            )
        self.scenario = scenario
        self.rng = rng if rng is not None else np.random.default_rng()
        self.brain = brain
        self.agents: dict[int, Agent] = {}
        self._ids = ids if ids is not None else IdAllocator()
        self._slots = slots if slots is not None else TargetSlots(scenario.targets)
        self._spawn_accumulator = 0.0
        # "uniform20" is the only personality_mix the plan currently defines; anything else
        # degrades to a single personality rather than guessing what was meant.
        self.n_personalities = 20 if scenario.agents.personality_mix == "uniform20" else 1

    @classmethod
    def from_scenario(
        cls,
        scenario,
        rng: np.random.Generator | None = None,
        brain: str = "brain",
        slots: TargetSlots | None = None,
        ids: IdAllocator | None = None,
    ) -> "Population":
        return cls(scenario, rng=rng, brain=brain, slots=slots, ids=ids)

    # -- queries --------------------------------------------------------

    @property
    def slots_used(self) -> list[int]:
        """Read-only view of the (possibly shared) slot occupancy, kept as a plain attribute name
        for anything already reaching into `pop.slots_used` -- `self._slots` is the real owner."""
        return self._slots.used

    @property
    def _next_id(self) -> int:
        """The next id this population's (possibly shared) allocator will hand out."""
        return self._ids._next

    def ids(self) -> list[int]:
        return list(self.agents.keys())

    def walking_ids(self) -> list[int]:
        return [i for i, a in self.agents.items() if a.status == "walking"]

    def positions(self, ids: list[int] | None = None) -> np.ndarray:
        keys = ids if ids is not None else self.agents.keys()
        if not keys:
            return np.zeros((0, 2))
        return np.array([self.agents[i].position for i in keys], float)

    # -- lifecycle --------------------------------------------------------

    def spawn(self, dt: float) -> list[int]:
        """Add new agents at the scenario's entry points, at `rate_per_s`, capped at
        `population_cap`. Returns the ids spawned this call (may be more than one for a large dt,
        or none if the cap is already full). The cap applies per cohort -- two `Population`s built
        via `build_cohorts` each spawn up to the scenario's own `population_cap`, since that's what
        "a standard crowd model running in the same space" (the plan's own words) means: two full
        crowds, not one crowd split between two brains."""
        sc = self.scenario
        self._spawn_accumulator += sc.spawn.rate_per_s * dt
        new_ids: list[int] = []
        while self._spawn_accumulator >= 1.0 and len(self.agents) < sc.spawn.population_cap:
            self._spawn_accumulator -= 1.0
            source_idx = int(self.rng.integers(len(sc.spawn.sources)))
            source = sc.spawn.sources[source_idx]
            preferred = sc.spawn.source_targets[source_idx] if sc.spawn.source_targets else None
            agent = Agent(
                id=self._ids.next(),
                position=np.array(source, float),
                personality=int(self.rng.integers(self.n_personalities)),
                brain=self.brain,
                preferred_target=preferred,
            )
            self.agents[agent.id] = agent
            new_ids.append(agent.id)
        # If the cap is what's stopping spawns (not a lack of budget), don't bank whatever's left
        # over. Any leftover here was earned partly -- or, over a long capped stretch, entirely --
        # during time the population was already full, so it isn't a genuine arrival waiting for
        # room; keeping any of it (even a fraction below 1.0) means the very next tick's own tiny
        # addition can tip it back over the threshold and fire an instant spawn regardless of that
        # tick's dt, refilling the cap in one step the moment a single slot frees up instead of
        # resuming arrivals at the configured rate. Dropping it to zero means a freed slot starts
        # the rate-timer over from scratch, same as a fly turned away at a full door rather than
        # one invisibly queued outside.
        if len(self.agents) >= sc.spawn.population_cap:
            self._spawn_accumulator = 0.0
        return new_ids

    def set_positions(self, ids: list[int], positions) -> None:
        """Sync positions after `world.physics.step()` has moved the walking agents. Feeding
        agents are parked at their slot and aren't expected here, but are silently ignored if
        passed (rather than fought over) since a stray update shouldn't corrupt state."""
        positions = np.asarray(positions, float).reshape(-1, 2)
        for i, p in zip(ids, positions):
            a = self.agents.get(i)
            if a is not None and a.status == "walking":
                a.position = p

    def feed(
        self,
        dt: float,
        capture_radius: float = DEFAULT_CAPTURE_RADIUS_M,
        assigned_targets: dict[int, int] | None = None,
        target_positions=None,
    ) -> list[int]:
        """Start feeding for any walking agent within `capture_radius` of a target that still has
        a free slot, then advance every currently-feeding agent's timer by `dt`. Returns the ids
        that started feeding this call. Slot occupancy is checked against `self._slots`, which may
        be shared with another `Population` walking the same venue -- so a target that another
        cohort's flies have filled up correctly turns this cohort's flies away too.

        `assigned_targets` (id -> target index, typically a brain's own `TargetAssignment.assigned`
        dict, passed in by `Sim.tick`) restricts each agent to capture only at *its own* assigned
        target -- an id absent from it falls back to the old "any target in range wins" check.
        This matters whenever a spawn source sits within `capture_radius` of a target that isn't
        the one that agent is actually walking to (see `SpawnConfig.source_targets`'s docstring):
        real bug, found from the user's own report ("they are spawning right on the sugar") --
        fixing *where a fly walks* (the `source_targets`/`preferred_target` work) wasn't enough on
        its own, because this method used to grab any walking agent near *any* target regardless of
        which one it was actually headed for, so a fly born next to the wrong target got scooped up
        before it ever took a step. Without `assigned_targets` (e.g. tests that drive `Population`
        directly with no brain in the loop) the old any-target-in-range behaviour is unchanged.

        `target_positions` (sequence indexable by target index, e.g. a list or `(n_targets, 2)`
        array of `(x, y)` pairs) overrides where each target *currently* is for the capture-radius
        check only. `Scenario`/`TargetConfig` are frozen dataclasses -- `sc.targets[ti].position`
        can never change once the scenario loads -- but a target moving mid-run is a real feature
        landing alongside this one, and the capture check has to compare against where the target
        actually is *now*, not the stale spot it started at: otherwise a fly could get captured at
        a target's old location after it moved away, or never get captured at the new one. Nothing
        else about a target (`kind`, `feeding_time_s`, its slot count via `self._slots`) is live --
        those stay config, read from `sc.targets[ti]` as always -- only the position used in the
        distance comparison is swapped. Leaving `target_positions` as `None` (the default -- every
        existing call site, every existing test, anything driving `Population` directly with no
        `Sim`/brain/moving-target concept in the loop) keeps today's exact behaviour: the position
        comes from `sc.targets[ti].position`, unchanged."""
        sc = self.scenario
        started: list[int] = []
        for a in self.agents.values():
            if a.status != "walking":
                continue
            own_target = assigned_targets.get(a.id) if assigned_targets is not None else None
            candidates = [(own_target, sc.targets[own_target])] if own_target is not None else list(enumerate(sc.targets))
            for ti, t in candidates:
                t_position = target_positions[ti] if target_positions is not None else t.position
                if np.linalg.norm(a.position - np.array(t_position, float)) <= capture_radius:
                    if self._slots.try_occupy(ti):
                        a.status = "feeding"
                        a.target_index = ti
                        a.feed_remaining_s = t.feeding_time_s
                        started.append(a.id)
                        break
                    # else: this target's full -- keep looking, there may be another in range
                    # (only possible when falling back to the no-assignment any-target check)
        for a in self.agents.values():
            if a.status == "feeding":
                a.feed_remaining_s -= dt
        return started

    def leave(self) -> list[int]:
        """Remove every agent that has finished feeding, freeing its slot. Returns their ids."""
        done = [i for i, a in self.agents.items() if a.status == "feeding" and a.feed_remaining_s <= 0]
        for i in done:
            a = self.agents.pop(i)
            self._slots.release(a.target_index)
        return done


def build_cohorts(scenario, rng: np.random.Generator | None = None) -> dict[str, "Population"]:
    """One `Population` per cohort sharing this scenario's venue: always a `"brain"` cohort (the
    fly-brained agents -- `ToyBrain` today, the real connectome once M1 lands), plus a `"baseline"`
    cohort too when the scenario's `[baseline] enabled = true`. Both cohorts share one `TargetSlots`
    (a target's feeding capacity is physical, not per-cohort) and one `IdAllocator` (so "agent 7"
    means the same fly regardless of which cohort's dict you look it up in) -- see this module's
    own docstring for why that sharing matters and how a combined tick is expected to use it.

    Both cohorts draw from the same `rng`, in cohort order each call -- deterministic for a given
    scenario seed and call order, same as a single `Population` already is, just now covering two
    cohorts' worth of draws from one stream instead of one cohort's worth."""
    rng = rng if rng is not None else np.random.default_rng()
    slots = TargetSlots(scenario.targets)
    ids = IdAllocator()
    cohorts = {"brain": Population(scenario, rng=rng, brain="brain", slots=slots, ids=ids)}
    if scenario.baseline.enabled:
        cohorts["baseline"] = Population(scenario, rng=rng, brain="baseline", slots=slots, ids=ids)
    return cohorts
