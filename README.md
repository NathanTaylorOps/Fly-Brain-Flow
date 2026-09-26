# Fly-Brain Flow

[![tests](https://github.com/NathanTaylorOps/Fly-Brain-Flow/actions/workflows/tests.yml/badge.svg)](https://github.com/NathanTaylorOps/Fly-Brain-Flow/actions/workflows/tests.yml)

*A crowd simulation where every agent is running a real fruit fly brain.*

![Two streams of agents crossing a corridor in the M0 playback viewer](assets/demo.png)

**Status:** M0 (plumbing) is done and independently reviewed twice — once for correctness, once for
quality — with 159 tests green. That's the map loader, scenario config, wall-aware odour, wall-bending
wind, the 2D collision solver, the spawn/feed/leave agent lifecycle, the baseline steering model, the
toy brain, the sim loop, the recorder, and the playback viewer above. **M1 (the real connectome) is
underway**, scoped in [docs/M1_PLAN.md](docs/M1_PLAN.md) — the MaleCNS dataset is pinned and
verified against neuPrint; the spiking-graph scaffold that runs on it is next.

## What

Scientists have mapped the complete nervous system of a fruit fly — 166,691 neurons and every connection between them — and put it online for free. I'm using it as the brain for a crowd: hundreds to thousands of simulated flies, each running the real wiring as its own live simulation, walking around a venue or a street map looking for food. Then I check whether the crowd behaves like real crowds and real traffic do — lanes forming, doorways clogging, phantom jams — against a standard crowd model running in the same space. And I can click on any fly and see what its brain is doing.

Two ideas carry most of the design. The inspector is essentially a replay debugger for agents: record what each fly sensed, recompute any one brain on demand. And most of the engineering is about making thousands of real brains fit on one rented GPU.

## Why

To find out whether a real insect brain, used as a navigation controller and scaled to a crowd, does anything the simple models don't. If the answer is no, that's the result.

## The plan

It's long and deliberate, on purpose: **[docs/PLAN.md](docs/PLAN.md)**. Scope, data, the brain, what flies can sense, how the brain moves the body, the world, the inspector, where it runs, how I'll know if it works, licences, open questions, risks.

## Milestones

- [x] **M0 — Plumbing.** Maps, physics, spawn/feed/leave, smell and wind, baseline agents, recorder, viewer — all on a toy brain. Tests pass.
- [ ] **M1 — One real fly** finds a moved target. Calibration gate passed.
- [ ] **M1.5 — Benchmark.** A measured number: how many full brains one GPU can run.
- [ ] **M2 — Tens of flies.** Personalities, queuing, the 22-car ring-road test.
- [ ] **M3 — Click any fly** in a recorded run and see its brain.
- [ ] **M4 — The result.** Lane formation and the doorway test vs baseline, with numbers.
- [ ] **M5 — Stretch.** Compare against a real dataset.

**Next up:** M1 — the real connectome.

## Running it

```
pip install -e '.[dev,brain]'
pytest

python scripts/record_demo.py --duration 60
python -m http.server -d viewer 8000
# open http://localhost:8000/?data=run.json
```

That runs the two-stream corridor scenario (the screenshot above), records it, and exports it for
`viewer/index.html` — a self-contained 2D top-down playback (no build step, no CDN): dots per
agent, colour-coded fly-brained vs. baseline, target(s) marked, venue walls drawn from the map's
own geometry. It's deliberately not the eventual live 3D neuron viewer the plan describes — that's
M3 scope, once a real brain exists to click into. This is the M0 version: proof a run can be
recorded and watched.

### Module map

| Piece | Lives in | What it does |
|---|---|---|
| Maps | `flybrainflow/world/map.py`, `scenarios/` | Built from a generator, an SVG/DXF floor plan, or a GeoJSON road file — every loader produces the same walkable grid. |
| Odour | `flybrainflow/world/fields.py` | Movable, per-target, wall-aware plumes. |
| Odour routing | `flybrainflow/world/geodesic.py` | Wall-aware shortest-path routing for odour and steering alike. |
| Wind | `flybrainflow/world/airflow.py` | Bends around obstacles and speeds up through gaps (potential flow, not real turbulence). |
| Physics | `flybrainflow/world/physics.py` | The 2D circle-collision solver that actually moves agents — one solver, fly-brained and baseline alike. |
| Agent lifecycle | `flybrainflow/agents.py` | Spawn (up to a population cap), feed at a target until its slot frees up, leave — open-boundary scenarios; `build_cohorts()` sets up a fly-brained and a baseline `Population` sharing one venue for a fair comparison. |
| Baseline brain | `flybrainflow/brains/baseline.py` | The standard-crowd-model reference: goal-seeking along the wall-aware geodesic field, social-force-style repulsion. |
| Toy brain | `flybrainflow/brains/toy.py` | A small fake-neuron stand-in for the real 166k-neuron connectome, so the surrounding plumbing gets debugged before the real dataset lands. |
| Sim loop | `flybrainflow/sim.py` | Ties it all into one runnable tick: spawn, steer (per cohort, through its own brain), move (one shared physics call), feed, leave. |
| Recorder | `flybrainflow/recorder.py` | Captures each tick to `.npz` — every agent's position, and, for fly-brained agents, the exact sensory input its brain saw that tick (the "replay debugger for agents" design). |
| Viewer export | `flybrainflow/viewer_export.py` | Turns a recording into the JSON `viewer/index.html` plays back. |

Both brains move through the same `physics.step()`, so the eventual fly-brain-vs-baseline comparison is of steering decisions, not of two different simulators.

## Licence and credit

Code is MIT. The brain data is MaleCNS (CC-BY, Janelia / Cambridge / MRC LMB / Google Research). Everything I've built on is listed in [ATTRIBUTION.md](ATTRIBUTION.md).

Planned and drafted with AI assistance; the decisions are mine.
