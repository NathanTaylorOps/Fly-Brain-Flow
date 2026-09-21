# Fly-Brain Flow

[![tests](https://github.com/NathanTaylorOps/Fly-Brain-Flow/actions/workflows/tests.yml/badge.svg)](https://github.com/NathanTaylorOps/Fly-Brain-Flow/actions/workflows/tests.yml)

*A crowd simulation where every agent is running a real fruit fly brain.*

**Status:** M0 (plumbing) in progress · map loader, scenario config, wall-aware odour, wall-bending wind, the 2D collision solver, the spawn/feed/leave agent lifecycle, the baseline steering model and the toy brain done, 87 tests green · **nothing runs end-to-end yet**

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

Nothing runs end-to-end yet. What exists can be tested:

```
pip install -e '.[dev]'
pytest
```

Maps can be built from a generator, an SVG or DXF floor plan, or a GeoJSON road file — see `flybrainflow/world/` and `scenarios/`. Odour plumes (movable, per-target, wall-aware) live in `flybrainflow/world/fields.py`; wall-routing for odour is `geodesic.py`; wind that actually bends around obstacles and speeds up through gaps (potential flow, not real turbulence) is `airflow.py`; the 2D circle-collision solver that actually moves agents — the same one for fly-brained and baseline agents alike — is `physics.py`. The agent lifecycle (spawning at entry points up to a population cap, feeding at a target until its slot frees up, leaving) lives in `flybrainflow/agents.py`, for open-boundary scenarios only — the closed/ring-road population setup is a separate M2 task. The standard-crowd-model reference this project measures the real fly brain against — goal-seeking along the wall-aware geodesic field, with social-force-style pedestrian and wall repulsion — is `flybrainflow/brains/baseline.py`. The toy brain — a small fake-neuron stand-in for the real 166k-neuron connectome, built so the surrounding plumbing gets debugged before the real dataset lands — is `flybrainflow/brains/toy.py`; both brains move through the same `physics.step()`, so the eventual comparison is of steering decisions, not of two different simulators.

## Licence and credit

Code is MIT. The brain data is MaleCNS (CC-BY, Janelia / Cambridge / MRC LMB / Google Research). Everything I've built on is listed in [ATTRIBUTION.md](ATTRIBUTION.md).

Planned and drafted with AI assistance; the decisions are mine.
