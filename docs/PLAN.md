# Fly-Brain Flow — the plan

**Status:** planning done · M0 (plumbing) not started · nothing runs yet
*Planned and drafted with AI assistance; the decisions are mine.*

This is the plan, in full. It's long because I'd rather find the mistakes on paper than in a half-built repo. The short version is in the [README](../README.md).

---

## What this is

Scientists have finished mapping the complete nervous system of a fruit fly — every neuron (166,691 of them) and every connection between them — and put it online for free. That's a real brain, wiring diagram and all, sitting in a download link.

I want to use it as the brain for a crowd.

The idea: put a few hundred to a few thousand simulated flies into a venue or a street map, give each one the real brain running as its own live simulation, give them something to want (sugar water, or a mate), and let them walk. Each fly feeds when it gets there, then vanishes; new ones keep arriving. Then I watch what the crowd does and check whether it behaves the way real crowds and real traffic behave — lanes forming, doorways clogging, phantom traffic jams — and compare it against a standard crowd model running in exactly the same space.

And at any point I can click on any single fly and see what its brain is doing, right now, and what it was doing along the way.

Two ideas carry most of the design. The inspector is essentially a replay debugger for agents: record what each fly sensed, recompute any one brain on demand, exactly. And most of the engineering is about making thousands of real brains fit on one rented GPU.

This is a spare-time project. I run operations for a building company by day. It's here because it's interesting and because I want to learn the stack properly.

## What I'm trying to find out

One question: **does a real insect brain, used as a navigation controller and scaled up to a crowd, produce realistic flow — and if so, what does that tell us?**

Plenty of simpler models already produce realistic-looking crowds (boids, social force, ant colony stuff). So the bar isn't "does it look like a crowd." The bar is "does the real brain do anything the simple models don't." If the answer is no, that's a result and I'll write it up as one.

## Ground rules

- **Core scope:** flies as pedestrians in a venue. That's the project.
- **Stretch scope, optional, first to be cut if needed:** flies as cars on a road (same brain, car body), and a "mate" target as a second thing to want.
- **Budget:** low. Free tiers first, cheap rented GPU hours when needed, hard spending caps set on every cloud account before anything runs.
- **Pace:** as fast as the work allows, no deadline. First targets: M0, then the benchmark.
- **Way of working:** plan properly, then build. Check-ins are informal; the milestone list further down is how I'll know whether anything's moving.

---

## The data

**Dataset: MaleCNS** — the complete male fruit fly nervous system, brain and nerve cord together. Released in 2025 (as a preprint; peer review still in progress) by Janelia (HHMI), University of Cambridge, MRC LMB and Google Research. Licence is CC-BY, meaning I can use it for anything as long as I credit them. It's live on neuPrint with a Python client and bulk downloads.

Why this one:
- The licence is clean (the other big fly brain dataset, FlyWire, is non-commercial-only).
- It's the *whole* nervous system, not just the brain.
- It includes the male courtship circuits, which I need if the mate target ever gets built.
- Google was involved, which was the original thread I pulled on.

Rules I'm holding myself to:
- **Pin the version.** The dataset is still being revised. I'll lock a version and file hashes on day one so results don't shift under me.
- **Bulk download, don't query per neuron.** 166k neurons through an API is a bad afternoon.
- **FlyWire is used once, for calibration only** (there's a published result on it I need to reproduce before trusting my own build). It never ships with the project.
- **Borrow code, not data.** There's an open-source project that already bolted the FlyWire brain onto a simulated fly body. Their code is MIT-licensed and useful. It's built on FlyWire, so anything data-shaped that comes out of it is non-commercial-only. Code only.
- **No data on my laptop.** It lives in the cloud. My machine is an editor.

If MaleCNS turns out to be a problem, the fallback is BANC — the female equivalent, published in *Nature* in 2026.

---

## The brain

Every fly runs the **full real connectome**. No cut-down "close enough" controller anywhere. That's the whole point.

There are two ways to simulate a brain like this, and I'm not picking one on paper:

- **Spiking** — models individual electrical spikes at sub-millisecond resolution. Expensive per step, but the fly brain is mostly quiet, and spiking simulators only do work when something fires. There's a published version of this (Shiu et al. 2024, *Nature*) that correctly predicted real fly feeding behaviour from the wiring alone.
- **Rate-based** — models each neuron's activity as a smooth level rather than individual spikes. Bigger time steps, but every connection gets computed every step whether anything's happening or not.

Which is cheaper depends on the implementation. So both get built (batched across agents so the GPU does many flies at once) and both get timed on the same hardware. The numbers decide.

Decisions that hold either way:

- **One shared weight matrix for all flies.** Each fly carries only its own activity state, its random seed and its personality. Thousands of separate 1 GB brains would be terabytes; this is the trick that makes many flies fit on one GPU.
- **No learning.** Flies leave after they feed, so anything they learned would never be used. Simpler.
- **Sensory input is done the simple way:** pick the real sensory neurons for a given stimulus and drive them harder the stronger the stimulus. That's exactly how the published model did it, and it worked. Fancier biology is an upgrade path, not a starting point.
- **Everything is reproducible.** Same seed, same run, every time. This matters later (see "watching a fly think").
- **Guard rails on activity from day one.** The excitatory/inhibitory labels on connections are machine-predicted, not certain. A few wrong signs across tens of millions of connections can make a brain run away or go dead, and a small test won't show it. So activity gets bounded from the start.
- **The clock ratio is a setting, not an accident.** A fly reacts in tens of milliseconds; a person walking takes several times longer (rough figures — the point is the gap, not the decimals). If I scale fly walking speed up to human speed but leave its reaction time alone, I get superhuman pedestrians who never bump into anything. How fast the brain runs relative to the world is an explicit dial, and turning it is part of the experiment.
- **"Happiness" and "frustration" are a display layer, and labelled as one.** There are no emotion neurons. What there are: reward neurons that fire when a fly feeds, hunger neurons, and arousal neurons. The inspector will show those and call them what they are.

---

## The flies

**Twenty personalities.** Not twenty different brains — one brain, twenty different settings applied at birth. Six dials:

| Dial | What it changes |
|---|---|
| Sugar sensitivity | how hard food pulls |
| Mate sensitivity | how hard a mate pulls |
| Timidity | how wide a berth it gives other flies |
| Boldness | pushes through a crowd vs hangs back |
| Baseline speed | fast walker or slow one |
| Noise | steady or erratic |

Plus a preference: food-leaning, mate-leaning, or mixed. A handful of archetypes (hungry-and-bold, hungry-and-timid, mate-seeking, slow-and-steady, erratic) and the rest are blends. It's a 20-row table.

Each fly is its own live simulation from that starting point. Two flies with the same personality will still behave differently because they see different things, start in different places, and have different seeds.

**Arriving.** Flies spawn at entry points at a rate I set, up to a population cap I set. Density isn't a dial directly — it comes out of arrival rate versus exit rate — so the cap is how you pin it.

**Leaving.** A fly reaches a target, takes a slot, feeds for two seconds (adjustable), and disappears. Same for mate targets.

**Targets.** Any number of them, placed anywhere, movable while the sim runs. Each has six slots (adjustable). Which gives a hard number worth designing around:

> **Max exit rate = targets × slots ÷ feeding time.** Six slots at two seconds = three flies a second per target. Spawn faster than the total and the queue grows forever.

That threshold is where the interesting crowd behaviour lives, so the sliders are built around it.

**When the slots are full,** nothing special happens. The smell is still there, so flies keep pressing in and mill around the edge — which is what real flies do at food. The physics keeps it from turning into a pile. I'll watch for permanent jams and only add a rule if one shows up.

**No flying.** Ever. Enforced in two places — see the next two sections.

---

## What a fly can sense

Each sense maps to real, named neurons in the dataset. The world computes the stimulus; the sim drives those neurons proportionally.

| Sense | Neurons driven | Notes |
|---|---|---|
| Smell of food | Or42b olfactory neurons (the food/vinegar attraction channel) | strength = odour concentration at the fly's position |
| Taste of food | sweet-taste neurons (Gr64f / Gr5a) on feet and mouthparts | fires when the fly is in a slot; starts the feeding timer. Same neurons the published model validated |
| Smell of a mate | Or47b olfactory neurons (male attraction to female pheromone) | stretch scope |
| Sight of a mate | LC10 visual neurons | optional, decide later |
| Other flies and walls, left and right | LC16 visual neurons | these are the "something's coming at me, back up / turn away" neurons **in walking flies** |
| Wind direction and speed | Johnston's organ (the antennae) | feeds the circuit real flies use to track a smell upwind |
| Bumping into things | leg touch neurons | minimal in v1; physics handles the hard part |

**The one that's deliberately never driven:** LC4, LC6, LPLC1, LPLC2. Those are the "something's coming at me, *jump*" neurons. Drive them and the brain tries to take off. Leaving them silent is half of how "no flying" is enforced.

---

## How the brain moves the body

The brain's output is a set of descending neurons — the ones that carry commands from brain to body. I read a few well-documented ones and turn them into speed and turning:

| Reads | Neurons | Does |
|---|---|---|
| Fast turning | DNa02 (left/right pair) | left minus right = turn rate. These are fed directly by the brain's compass circuit (PFL3) and its "is this smell good" circuit (MBON32) — the exact spot where wanting food becomes steering |
| Slow turning | DNa01 (left/right pair) | same idea, gentler and more sustained |
| Turning gain | PFL2 | how hard to steer, without a direction |
| Forward | DNp09 | walk speed; very high drive = stop (a documented quirk, which I'm using as the brake) |
| Backward | MDN ("moonwalker") | reverse; net speed = forward minus backward |
| Jump / take off | giant fibre pathway | **read for display, never acted on** — the other half of "no flying" |

Same bridge for both modes. Only the body limits change:

- **Pedestrian:** ~1.3 m/s cap, human-plausible turning, quarter-metre radius.
- **Car (stretch):** stuck to the road, car length and turning circle, and the "something's looming ahead" signal becomes braking. This is fly-brain-drives-a-car and I'll present it that way. The braking part is the bit that actually holds up biologically.

---

## The world

**Maps are swappable.** Two input formats, one loader:
- Street maps from OpenStreetMap (roads become drivable/walkable polygons).
- Venues as DXF or SVG — which means I can draw a floor plan in Chief Architect and export it straight in.

Both turn into the same thing internally: a walkable area, wall boundaries, and grids for smell and wind.

**Flies stay on the walkable bit.** Physics enforces it, not the brain.

**Smell travels through the geometry, not through walls.** Odour spreads along corridors, carried by the wind. If I did it as-the-crow-flies, every fly would walk into a wall trying to reach food on the other side. Recomputed whenever a target moves.

**Wind is a dial** — direction and speed, adjustable live. This turned out to be necessary, not decorative: real flies find smells by going upwind when they catch a whiff and casting sideways when they lose it, and the dataset has that circuit. No wind is a legitimate setting; it should navigate worse, and showing both is part of the result.

**Physics is a simple 2D circle-collision solver.** No legs, no articulated body — these are point agents at human or car scale. The **same** solver runs the baseline model, so comparisons are fair.

**Two boundary modes:** open (entry points and exits — the pedestrian scenarios) and closed (a ring, fixed population, no targets — the classic traffic-jam test).

**Every scenario is a config file** (map, spawn, targets, wind, clock ratio, personalities, seed, duration, mode). That's what "swappable venue" means in practice, and it means every run can be repeated exactly.

---

## Watching a fly think

Two ways to run it:

- **Live.** Full brains computing in real time; move targets, change the wind, adjust spawn, watch the crowd react. Agent count is limited by whatever the benchmark says one rented GPU can do — probably hundreds.
- **Recorded.** A full-brain run with thousands of flies computed offline (slower than real time is fine), then played back. Every fly still had its full brain; you just can't poke the world mid-playback.

**Click any fly, in either mode, and see its brain.** The trick that makes this cheap: because every fly is deterministic, all I record per fly is its path, its sensory inputs (a few hundred numbers per tick), its seed and its personality. Click one and I recompute *just that brain* from its recorded inputs — exactly what it computed the first time, with only one brain running live. Everything it saw, including other flies, is already in the recorded inputs, so the rest of the crowd doesn't need re-running. It's a replay debugger for agents.

Four layers, from glanceable to exhaustive:

1. **Its path**, coloured by what was driving it at each point — chasing food, chasing a mate, avoiding someone, feeding.
2. **A timeline of brain regions** (about eighty of them) lighting up along that path. This is the one that reads as "thinking over time."
3. **The whole brain in 3D** at any instant you scrub to — all 166k neurons.
4. **"Why did it turn here?"** — trace which sense changed, which pathway carried it, which motor neurons fired. There's an existing tool for this kind of pathway tracing.

Layers 1–2 are pre-stored for *every* fly (~200 KB per fly-lifetime), so they're instant. Layers 3–4 recompute on demand — seconds to a minute.

**Rendering:** the 166k-neuron view shows brain regions and cell bodies at the overview level and only loads individual neuron shapes when you zoom. Raw 166k dots is noise to a human, and my laptop won't draw it anyway.

**The viewer is a web page** (Three.js), so it runs in a browser on anything, and the recorded mode can be hosted for free as static files. No always-on GPU server; the full-resolution recompute either runs while I've got a GPU rented, or on a pay-per-second on-demand GPU.

---

## Where it runs

- **My laptop:** Surface Pro 9, 8 GB RAM, integrated graphics, not much disk. It's a keyboard and a screen. Nothing runs on it and nothing is stored on it.
- **Day-to-day dev:** GitHub Codespaces on this repo (free tier as of writing: 2 cores, 8 GB, no GPU, stops when idle). Editing, logic tests, CPU-side data filtering.
- **Free GPU:** Kaggle notebooks (as of writing: a T4, 30 hours a week, 12-hour sessions). Benchmark, calibration, recorded runs.
- **Also an option:** an Azure free account comes with a starting credit that would cover a few GPU hours — enough for the benchmark — *if* the trial subscription is allowed GPU quota, which needs checking first.
- **Paid GPU, only if needed:** RunPod or Vast.ai by the hour. Spending caps set before the first hour.
- **A toy brain for local work.** A few hundred fake neurons running through the *same code path* as the real thing. All the plumbing — spawning, feeding, smell, walls, the inspector — gets built and debugged against the toy for free. The real 166k brain only comes out for validation and scale. The toy won't catch things that only break at scale (memory, numerical drift, streaming bandwidth); short, cheap paid-GPU checkpoints catch those.
- **Stack:** Python and PyTorch for the sim, NumPy for the toy and the baseline, custom 2D physics, Three.js viewer, WebSocket for live mode, Parquet/NPZ recordings, TOML scenarios, osmnx and DXF/SVG loaders. Recordings live on a Kaggle dataset or a free object store, not in git.

---

## How I'll know if it works

**A baseline from day one.** Before any real brain runs, a standard social-force crowd model (forked from PySocialForce, MIT) is running in the same engine, same physics, same maps, same measurements. If the comparison gets built at the end it'll be apples to oranges and twice the work.

**A calibration gate before scaling.** The published spiking model reproduced real feeding behaviour from the wiring. I reproduce that result with my own build — on the same dataset they used — before I trust my build on anything. Then I switch datasets and check again. One variable at a time, so if it breaks I know which thing broke it.

**A benchmark before any schedule.** Both brain types, batched, one GPU, timed at a hundred and at five hundred flies. Every cost estimate before that number exists is a guess and gets treated as one. If the number is bad, the fallback is already agreed: live mode runs tens to hundreds, recorded mode absorbs the rest.

**Named phenomena, pass or fail** — not "does it look right":
- *Pedestrians:* do lanes form on their own when two streams walk through each other? Does a doorway clog worse when everyone's in a hurry? (Both are well-documented real effects.)
- *Cars:* the Sugiyama experiment — 22 cars on a ring road with nothing in the way, and a traffic jam appears out of nowhere and travels backwards. Famous, clean, and either I reproduce it or I don't.

Note that all of these show up at **tens to hundreds** of agents. Thousands is for the spectacle; the findings don't need it. The ring-road test runs at 22 flies the moment one fly works.

**Measured the same way for both models:** flow vs density (the standard crowd curve), a lane-formation index, throughput per target, time-to-target, queue length, overlap count.

**Tests:** same seed gives the same run; no fly is ever outside the walkable area; arrivals minus departures equals current population; every module has toy-brain unit tests; brain activity stays bounded.

### Milestones

| | Done when | Costs |
|---|---|---|
| **M0 — Plumbing** | Maps load, physics works, flies spawn/feed/leave, smell and wind fields exist, baseline agents walk, recorder and viewer run — all on the toy brain. Tests pass. | nothing |
| **M1 — One real fly** | Calibration gate passed. One full-brain fly finds a moved target with a believable searching path with wind, and a visibly worse one without. Inspector layers 1–3 work. | Kaggle |
| **M1.5 — Benchmark** | Real numbers. Brain type chosen. Live-mode fly ceiling known. | an hour or two of GPU |
| **M2 — Tens of flies** | Personalities, both target types, slot queuing, no overlaps. Ring-road test at 22 cars, result recorded either way. | cheap |
| **M3 — Inspect at scale** | Click any fly in a recorded run, get all four layers. | cheap |
| **M4 — The actual result** | Lane formation and the doorway test at hundreds of flies vs baseline, with numbers. A recorded run at thousands. | bounded paid GPU |
| **M5 — Stretch** | Compare against a real pedestrian or traffic dataset with an actual number. | optional |

---

## Licences and credit

- This code: MIT.
- MaleCNS data: CC-BY. Credit to Berg et al. 2025 and the Janelia MaleCNS project.
- FlyWire (calibration only, never shipped): CC-BY-NC. Credit to Dorkenwald et al. and Schlegel et al. 2024.
- Code I've leaned on: Shiu et al.'s brain model, the community embodied-fly project (code only), PySocialForce, flyvis, Connectome Interpreter. All in [`ATTRIBUTION.md`](../ATTRIBUTION.md).
- **This project doesn't get sold** while anything derived from non-commercial data exists anywhere in the pipeline. Saying it now so "personal project" can't quietly become "product" later.

---

## Still to decide

| | What | Needed by |
|---|---|---|
| 1 | Repo name (working title: `fly-brain-flow`) | first commit |
| 2 | Benchmark result | brain type, everything downstream |
| 3 | Confirm in the pinned dataset: synapse count, optic lobes present, every neuron type listed above actually annotated | M1 |
| 4 | Dedicated "stop" neurons (v1 uses the forward-drive quirk) | M2 |
| 5 | Which neurons stand in for "frustration" | inspector polish |
| 6 | What happens when slots are full — watch first, add a rule only if needed | M2 |
| 7 | Dataset version tag and hashes | first data pull |
| 8 | Hosting for the on-demand full-brain recompute | M3 |
| 9 | Include the visual mate-tracking channel, or smell only | M2 |
| 10 | Default clock ratio per mode | M1 |
| 11 | Do flies ever come back with memory? (default: no) | only if a reason appears |
| 12 | The actual 20-row personality table | M2 |
| 13 | Slot layout at a target (ring of six vs an area) | M2 |

---

## What could go wrong

| Risk | Odds | Hurts | Plan |
|---|---|---|---|
| It never finishes — open-ended, solo, spare time | High | Everything | The milestone list. Stretch scope is labelled so cutting it isn't failure. M0 and the benchmark first. |
| Neither brain type runs enough flies on cheap GPUs | Medium | High | Fallback agreed above; the findings live at tens–hundreds anyway |
| Calibration gate fails | Medium | High | One variable at a time tells me whether it's my code or the dataset |
| Brain runs away or goes dead at scale (wrong connection signs) | Medium | Medium | Guard rails from day one; periodic real-GPU checks |
| Flies can't find the food | Medium | High | Wind + the real upwind circuit is the correct mechanism; tested with and without at M1 |
| Sparse GPU maths runs far below spec | High | Medium | Expected; measured, not assumed; sharing one weight matrix is the main lever |
| Dataset gets revised under me | Low | Medium | Version pinned |
| Cloud bill creeps | Low | Medium | Caps before first use |
| Car mode looks like a gimmick and cheapens the real result | Medium | Low | Present it honestly; lead with the braking mapping, which does hold up |
| Mate target ends up just "second sugar target" | Medium | Low | First thing to cut |
| Wall-aware odour (geodesic distance around barriers, needed for a stadium/station venue) is plain Dijkstra — correct, not fast; recomputing it every frame for a live-dragged target on a full-size map could be too slow | Medium | Medium | Only recomputed when a target's position actually changes, not every step; benchmark on the real venue size before relying on live-drag at scale, swap in a vectorized wavefront approximation if it's too slow |
| The wind model (potential flow) gets deflection and gap speed-up right but has no turbulence, eddies, or wake recirculation — a real pillar sheds a churning vortex behind it that this can never show | High | Low–Medium | Documented as a permanent limitation, not a bug to eventually fix; matters only if a specific result turns out to hinge on turbulent behaviour, which would need real CFD and is out of scope |
| Wind only flows through venue geometry where an opening is explicitly declared (`open_ends` or similar) — a floor plan with no modelled doors/gaps will show zero wind, correctly but unhelpfully | Medium | Medium | Every real venue (stadium/station) needs its actual entrances/gates traced as openings when the map is built, not assumed; a venue with no declared openings needs to visibly say so rather than silently return zero |
| ~~`agents.py`'s `Population` has no notion of "which brain drives this agent"~~ — **resolved**: `Agent.brain` tags each agent's cohort, and `agents.build_cohorts()` hands back one `Population` per cohort sharing a `TargetSlots` (so a target's feeding capacity is physical, not double-booked per cohort) and an `IdAllocator` (so ids stay unique across cohorts). What's still open: the sim loop itself (`flybrainflow.sim`, not started) hasn't been written yet, so nothing has actually driven two cohorts through one shared `physics.step()` call end to end — that's the real proof this design works, not just that the bookkeeping doesn't collide | Low | Medium | Prove it out when the sim loop is built: one combined `physics.step()` call over every cohort's ids each tick, each cohort's own brain producing its slice of the velocities |

---

## Scenario settings

What a scenario file looks like. Everything under `[wind]`, `[spawn]` and `[[targets]]` is adjustable live in live mode.

```toml
[scenario]
name            = "corridor_bidirectional"
seed            = 42
duration_s      = 300
scale_mode      = "pedestrian"      # "pedestrian" | "car"
boundary_mode   = "open"            # "open" | "closed"
timescale_ratio = 1.0               # brain ms per world ms

[map]
source          = "venues/corridor.svg"   # or "osm:<bbox>"
walkable_grid_m = 0.1

[wind]
direction_deg = 90
speed_mps     = 0.5

[spawn]
sources        = [[0, 5], [40, 5]]
rate_per_s     = 1.5
population_cap = 200

[[targets]]
kind           = "sugar"            # "sugar" | "mate"
position       = [40, 5]
slots          = 6
feeding_time_s = 2.0
odor_strength  = 1.0
odor_range_m   = 30

[agents]
personality_mix   = "uniform20"
sensing_radius_m  = 3.0
field_of_view_deg = 270
radius_m          = 0.25
max_speed_mps     = 1.3

[baseline]
enabled = true                      # social-force agents in the same run
```

---

## Reading list

**The data**
- Berg et al. (2025). Sexual dimorphism in the complete connectome of the *Drosophila* male central nervous system. bioRxiv 2025.10.09.680999 — https://male-cns.janelia.org/
- Bates et al. (2026). Distributed control circuits across a brain-and-cord connectome. *Nature* — the female equivalent (BANC)
- Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634 — FlyWire, https://codex.flywire.ai/
- Schlegel et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634
- A tutorial covering all the fly datasets: https://github.com/sjcabs/fly_connectome_data_tutorial

**Simulating the brain**
- Shiu et al. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634 — code: https://github.com/philshiu/Drosophila_brain_model
- Lappalainen et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature* — code: https://github.com/TuragaLab/flyvis
- The community embodied-fly project (code reference): https://github.com/erojasoficial-byte/fly-brain
- A fast Apple-silicon port of the spiking model: https://github.com/Kisame76/drosophila-brain-mlx
- Connectome Interpreter (pathway tracing): YijieYin/connectome-interpreter
- Dayan & Abbott, *Theoretical Neuroscience*, ch. 7 — firing-rate models

**Steering, walking, not-bumping-into-things**
- Neural circuit mechanisms for steering control in walking *Drosophila*. eLife 102230 — https://elifesciences.org/articles/102230
- Rayshubskiy et al. Fine-grained descending control of steering in walking *Drosophila* — https://pmc.ncbi.nlm.nih.gov/articles/PMC10614758/
- Bidaye et al. (2020). Two brain pathways initiate distinct forward walking programs. *Neuron*
- Bidaye et al. (2014). Neuronal control of *Drosophila* walking direction. *Science*
- Sen et al. (2017). Moonwalker descending neurons mediate visually evoked retreat. *Curr Biol* — https://www.sciencedirect.com/science/article/pii/S0960982217301446
- Wu et al. (2016). Visual projection neurons in the *Drosophila* lobula link feature detection to distinct behavioral programs. eLife — https://elifesciences.org/articles/21022
- Spatial readout of visual looming in the central brain of *Drosophila*. eLife 57685
- Cheong et al. (2025). Organization of circuits linking descending input to motor output in the *Drosophila* Male Adult Nerve Cord connectome. eLife 96084 — https://elifesciences.org/articles/96084
- Takemura et al. (2024). A connectome of the male *Drosophila* ventral nerve cord. eLife — the MANC dataset itself

**Smell and wind**
- Suver et al. (2019). Encoding of wind direction by central neurons in *Drosophila*. *Neuron* — https://pubmed.ncbi.nlm.nih.gov/30948249/
- Matheson et al. (2022). A neural circuit for wind-guided olfactory navigation. *Nat Commun* — https://www.nature.com/articles/s41467-022-32247-7
- Hulse et al. (2021). A connectome of the *Drosophila* central complex. eLife 66039
- Semmelhack & Wang (2009). Select *Drosophila* glomeruli mediate innate olfactory attraction and aversion. *Nature*
- Dweck et al. (2015). Pheromones mediating copulation and attraction in *Drosophila*. *PNAS*
- Stone et al. (2017). An anatomically constrained model for path integration in the bee brain. *Curr Biol*

**Crowds and traffic**
- Helbing & Molnár (1995). Social force model for pedestrian dynamics. *Phys Rev E* 51
- Helbing, Farkas & Vicsek (2000). Simulating dynamical features of escape panic. *Nature*
- Sugiyama et al. (2008). Traffic jams without bottlenecks. *New J Phys* 10 — the ring-road experiment
- Reynolds (1987), Boids. Vicsek et al. (1995), self-propelled particles.
- PySocialForce: https://github.com/yuxiang-gao/PySocialForce
- Datasets: ETH/UCY pedestrian trajectories; Stanford Drone Dataset; TrajNet++

---

## Jargon, decoded

- **Connectome** — the full wiring diagram of a nervous system: every neuron and every connection.
- **MaleCNS / BANC / FlyWire** — the three big fly datasets: male whole nervous system / female whole nervous system / female brain only.
- **Nerve cord (VNC)** — the fly's version of a spinal cord. Leg control lives here.
- **Descending neurons (DNs)** — the wires from brain to body. DNa01/DNa02 steer, DNp09 goes forward, MDN goes backward.
- **Central complex** — the fly's navigation centre. PFL2/PFL3 are its outputs that turn "where I'm facing vs where I want to go" into steering. hΔC is the part that combines wind and smell.
- **Mushroom body** — learning and "is this good or bad." Its output neurons (MBONs) and reward dopamine neurons (PAM) are what I'm using for the display-only "happiness."
- **Olfactory / gustatory receptor neurons** — smell and taste inputs. Or42b and Or47b are specific smell channels; Gr64f/Gr5a are sweet taste.
- **LC neurons** — visual feature detectors. LC16 says "back up." LC4/LC6/LPLC2 say "jump," which is why they're never driven.
- **Johnston's organ** — the antennae, used here as the wind sensor.
- **Spiking vs rate model** — simulating individual electrical pulses vs simulating smooth activity levels.
- **Direct injection** — driving named sensory neurons harder or softer to represent a stimulus.
- **Body bridge** — the small piece of code that turns descending-neuron activity into movement.
- **Live vs recorded** — computing everything in real time vs computing it once and playing it back, with click-to-inspect working in both.
- **Fundamental diagram** — flow against density; the standard curve for describing a crowd or a road.
