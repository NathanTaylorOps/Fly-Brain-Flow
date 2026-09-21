# Fly-Brain Flow

[![tests](https://github.com/NathanTaylorOps/Fly-Brain-Flow/actions/workflows/tests.yml/badge.svg)](https://github.com/NathanTaylorOps/Fly-Brain-Flow/actions/workflows/tests.yml)

*A crowd simulation where every agent is running a real fruit fly brain.*

**Status:** M0 (plumbing) in progress · map loader, scenario config, wall-aware odour, wall-bending wind, the 2D collision solver, the spawn/feed/leave agent lifecycle (fly-brained/baseline cohorts sharing one venue), the baseline steering model, the toy brain, the sim loop and the recorder done, 106 tests green · **a 2D playback viewer is the one M0 item left**

## What

Scientists have mapped the complete nervous system of a fruit fly — 166,691 neurons and every connection between them — and put it online for free. I'm using it as the brain for a crowd: hundreds to thousands of simulated flies, each running the real wiring as its own live simulation, walking around a venue or a street map looking for food. Then I check whether the crowd behaves like real crowds and real traffic do — lanes forming, doorways clogging, phantom jams — against a standard crowd model running in the same space. And I can click on any fly and see what its brain is doing.

Two ideas carry most of the design. The inspector is essentially a replay debugger for agents: record what each fly sensed, recompute any one brain on demand. And most of the engineering is about making thousands of real brains fit on one rented GPU.

## Why

To find out whether a real insect brain, used as a navigation controller and scaled to a crowd, does anything the simple models don't. If the answer is no, that's the result.

## The plan

It's long and deliberate, on purpose: **[docs/PLAN.md](docs/PLAN.md)**. Scope, data, the brain, what flies can sense, how the brain moves the body, the world, the inspector, where it runs, how I'll know if it works, licences, open questions, risks.

## Milestones

- [ ] **M0 — Plumbing.** Maps, physics, spawn/feed/leave, smell and wind, baseline agents, recorder, viewer — all on a toy brain. Tests pass.
- [ ] **M1 — One real fly** finds a moved target. Calibration gate passed.
- [ ] **M1.5 — Benchmark.** A measured number: how many full brains one GPU can run.
- [ ] **M2 — Tens of flies.** Personalities, queuing, the 22-car ring-road test.
- [ ] **M3 — Click any fly** in a recorded run and see its brain.
- [ ] **M4 — The result.** Lane formation and the doorway test vs baseline, with numbers.
- [ ] **M5 — Stretch.** Compare against a real dataset.

**Next up:** M0, then the benchmark.

## Running it

There's no viewer yet, so nothing produces a picture you can watch -- but the sim loop runs and a
run can be recorded to disk:

```
pip install -e '.[dev]'
pytest

python -c "
from flybrainflow.scenario import Scenario
from flybrainflow.sim import Sim
from flybrainflow.recorder import Recorder

sc = Scenario.from_dict({
    'scenario': {'name': 'demo', 'boundary_mode': 'open'},
    'map': {'source': 'gen:corridor?length=15&width=5'},
    'spawn': {'sources': [[1.0, 2.5]], 'rate_per_s': 2.0, 'population_cap': 8},
    'targets': [{'kind': 'sugar', 'position': [13.0, 2.5], 'slots': 4, 'feeding_time_s': 1.0}],
})
sim = Sim.from_scenario(sc)
rec = Recorder(sim)
for _ in range(300):
    sim.tick(0.1)
    rec.capture()
rec.save('demo_run.npz')
print({tag: len(pop.agents) for tag, pop in sim.cohorts.items()})
"
```

Maps can be built from a generator, an SVG or DXF floor plan, or a GeoJSON road file — see `flybrainflow/world/` and `scenarios/`. Odour plumes (movable, per-target, wall-aware) live in `flybrainflow/world/fields.py`; wall-routing for odour is `geodesic.py`; wind that actually bends around obstacles and speeds up through gaps (potential flow, not real turbulence) is `airflow.py`; the 2D circle-collision solver that actually moves agents — the same one for fly-brained and baseline agents alike — is `physics.py`. The agent lifecycle (spawning at entry points up to a population cap, feeding at a target until its slot frees up, leaving) lives in `flybrainflow/agents.py`, for open-boundary scenarios only — the closed/ring-road population setup is a separate M2 task. `agents.build_cohorts()` sets up a fly-brained and a baseline `Population` sharing one venue — same target slots, same id sequence — for the "standard crowd model running in the same space" comparison the plan calls for. The standard-crowd-model reference this project measures the real fly brain against — goal-seeking along the wall-aware geodesic field, with social-force-style pedestrian and wall repulsion — is `flybrainflow/brains/baseline.py`. The toy brain — a small fake-neuron stand-in for the real 166k-neuron connectome, built so the surrounding plumbing gets debugged before the real dataset lands — is `flybrainflow/brains/toy.py`; both brains move through the same `physics.step()`, so the eventual comparison is of steering decisions, not of two different simulators. `flybrainflow/sim.py`'s `Sim` ties all of that into one runnable tick — spawn, steer (per cohort, through its own brain), move (one shared physics call across every cohort), feed, leave. `flybrainflow/recorder.py`'s `Recorder` captures each tick to an `.npz` file — every agent's position, and, for fly-brained agents specifically, the exact 6-channel sensory input its brain saw that tick, per the plan's "replay debugger for agents" design.

## Licence and credit

Code is MIT. The brain data is MaleCNS (CC-BY, Janelia / Cambridge / MRC LMB / Google Research). Everything I've built on is listed in [ATTRIBUTION.md](ATTRIBUTION.md).

Planned and drafted with AI assistance; the decisions are mine.
