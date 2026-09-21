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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_CAPTURE_RADIUS_M = 0.5


@dataclass
class Agent:
    id: int
    position: np.ndarray
    personality: int
    status: str = "walking"  # "walking" or "feeding"
    target_index: int | None = None
    feed_remaining_s: float = 0.0


class Population:
    """Tracks every live agent in an open-boundary scenario: who's walking, who's feeding where,
    and how many free slots each target has left."""

    def __init__(self, scenario, rng: np.random.Generator | None = None):
        if scenario.meta.boundary_mode != "open":
            raise ValueError(
                f"Population is for open-boundary scenarios; '{scenario.meta.name}' is "
                f"boundary_mode={scenario.meta.boundary_mode!r} (closed-mode population setup "
                "is an M2 task, tied to the venue's own geometry, not built here)"
            )
        self.scenario = scenario
        self.rng = rng if rng is not None else np.random.default_rng()
        self.agents: dict[int, Agent] = {}
        self._next_id = 0
        self._spawn_accumulator = 0.0
        self.slots_used = [0] * len(scenario.targets)
        # "uniform20" is the only personality_mix the plan currently defines; anything else
        # degrades to a single personality rather than guessing what was meant.
        self.n_personalities = 20 if scenario.agents.personality_mix == "uniform20" else 1

    @classmethod
    def from_scenario(cls, scenario, rng: np.random.Generator | None = None) -> "Population":
        return cls(scenario, rng=rng)

    # -- queries --------------------------------------------------------

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
        or none if the cap is already full)."""
        sc = self.scenario
        self._spawn_accumulator += sc.spawn.rate_per_s * dt
        new_ids: list[int] = []
        while self._spawn_accumulator >= 1.0 and len(self.agents) < sc.spawn.population_cap:
            self._spawn_accumulator -= 1.0
            source = sc.spawn.sources[self.rng.integers(len(sc.spawn.sources))]
            agent = Agent(
                id=self._next_id,
                position=np.array(source, float),
                personality=int(self.rng.integers(self.n_personalities)),
            )
            self.agents[agent.id] = agent
            new_ids.append(agent.id)
            self._next_id += 1
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

    def feed(self, dt: float, capture_radius: float = DEFAULT_CAPTURE_RADIUS_M) -> list[int]:
        """Start feeding for any walking agent within `capture_radius` of a target that still has
        a free slot (first match wins if more than one target is in range), then advance every
        currently-feeding agent's timer by `dt`. Returns the ids that started feeding this call.
        """
        sc = self.scenario
        started: list[int] = []
        for a in self.agents.values():
            if a.status != "walking":
                continue
            for ti, t in enumerate(sc.targets):
                if self.slots_used[ti] >= t.slots:
                    continue
                if np.linalg.norm(a.position - np.array(t.position, float)) <= capture_radius:
                    a.status = "feeding"
                    a.target_index = ti
                    a.feed_remaining_s = t.feeding_time_s
                    self.slots_used[ti] += 1
                    started.append(a.id)
                    break
        for a in self.agents.values():
            if a.status == "feeding":
                a.feed_remaining_s -= dt
        return started

    def leave(self) -> list[int]:
        """Remove every agent that has finished feeding, freeing its slot. Returns their ids."""
        done = [i for i, a in self.agents.items() if a.status == "feeding" and a.feed_remaining_s <= 0]
        for i in done:
            a = self.agents.pop(i)
            self.slots_used[a.target_index] -= 1
        return done
