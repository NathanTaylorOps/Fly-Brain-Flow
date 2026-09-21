"""The recorder: capture enough per tick to play a run back later and, per the plan's own
"replay debugger for agents" design, to recompute any one fly's brain exactly from what it saw --
without re-running the rest of the crowd.

    sim = Sim.from_scenario(scenario)
    rec = Recorder(sim)
    for _ in range(n_ticks):
        sim.tick(dt)
        rec.capture()
    rec.save("runs/demo.npz")

    recording = Recorder.load("runs/demo.npz")
    recording["x"], recording["y"]     # every agent's position, every tick it was alive
    recording["sense"]                 # the toy/real brain's 6-channel sensory input, that tick

What's recorded, and why NPZ rather than Parquet for now: the plan names "Parquet/NPZ" as the
recording format, and `pyproject.toml` already lists `pyarrow` under the `data` extra for when
that matters (a real run, thousands of flies, many ticks, wanting columnar compression and
random access by fly). NPZ needs nothing beyond NumPy, which this whole project already depends
on, and every M0 run is small enough that a flat NPZ file is simpler and just as fast to write
and load correctly -- so that's what's actually implemented here. Swapping the storage format
later doesn't change anything upstream: `Recorder.capture()`'s in-memory frames and `load()`'s
returned arrays are the real interface either way.

One row per (tick, cohort, agent) that was alive that tick -- walking or feeding, not yet spawned
or already left. `sense` is only meaningful for agents whose brain actually exposes a
`last_sense` dict (`ToyBrain` does; `Baseline` doesn't have anything resembling a sensory channel
to record) -- rows for any other cohort get NaN there, not a fabricated value standing in for
"nothing to report."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SENSE_WIDTH = 6  # must match brains.toy._N_SENSE; not imported to avoid recorder<->brain coupling
# beyond the one thing that actually matters here (how many numbers a sense vector has).


class Recorder:
    """Captures one frame per `capture()` call -- call it once per tick, right after `sim.tick()`,
    while each brain's `last_sense` (if it has one) still reflects that tick and before the next
    tick overwrites it."""

    def __init__(self, sim) -> None:
        self.sim = sim
        self._rows: list[tuple] = []  # (t, id, brain, x, y, status, target_index, personality, sense)

    def capture(self) -> None:
        sim = self.sim
        t = sim.t
        for tag, pop in sim.cohorts.items():
            last_sense = getattr(sim.brains.get(tag), "last_sense", None)
            for i, a in pop.agents.items():
                sense = None if last_sense is None else last_sense.get(i)
                if sense is None:
                    sense = np.full(_SENSE_WIDTH, np.nan)
                self._rows.append(
                    (
                        t,
                        i,
                        tag,
                        float(a.position[0]),
                        float(a.position[1]),
                        a.status,
                        -1 if a.target_index is None else a.target_index,
                        a.personality,
                        np.asarray(sense, float),
                    )
                )

    def __len__(self) -> int:
        return len(self._rows)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """The recorded rows as one dict of columnar arrays -- what `save()` writes and `load()`
        hands back, so a caller that wants the data without touching disk (a test, an in-process
        inspector) can get exactly the same shape either way."""
        n = len(self._rows)
        if n == 0:
            return {
                "t": np.zeros(0),
                "id": np.zeros(0, dtype=np.int64),
                "brain": np.zeros(0, dtype="<U16"),
                "x": np.zeros(0),
                "y": np.zeros(0),
                "status": np.zeros(0, dtype="<U16"),
                "target_index": np.zeros(0, dtype=np.int64),
                "personality": np.zeros(0, dtype=np.int64),
                "sense": np.zeros((0, _SENSE_WIDTH)),
                "seed": np.array(self.sim.scenario.meta.seed),
                "name": np.array(self.sim.scenario.meta.name),
            }
        t, i, brain, x, y, status, target_index, personality, sense = zip(*self._rows)
        return {
            "t": np.array(t, dtype=float),
            "id": np.array(i, dtype=np.int64),
            "brain": np.array(brain, dtype="<U16"),
            "x": np.array(x, dtype=float),
            "y": np.array(y, dtype=float),
            "status": np.array(status, dtype="<U16"),
            "target_index": np.array(target_index, dtype=np.int64),
            "personality": np.array(personality, dtype=np.int64),
            "sense": np.stack(sense).astype(float),
            "seed": np.array(self.sim.scenario.meta.seed),
            "name": np.array(self.sim.scenario.meta.name),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_arrays())

    @classmethod
    def load(cls, path: str | Path) -> dict[str, np.ndarray]:
        """Returns the recorded columns as a plain dict of arrays -- not a live `Recorder` (there's
        no `Sim` to attach to once this is loaded back from disk), just the data. `data["seed"]`
        and `data["name"]` are 0-d arrays; use `.item()` to get the plain Python value back out."""
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}
