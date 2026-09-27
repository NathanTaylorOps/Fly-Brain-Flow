# M1 plan — one real fly

M0 built the plumbing on a fake brain. M1 replaces that fake brain with the real one, for exactly
one fly at a time, and proves it's trustworthy before it goes anywhere near a crowd.

## Done when (from docs/PLAN.md, unchanged)

- Calibration gate passed.
- One full-brain fly finds a moved target with a believable searching path with wind, and a
  visibly worse one without.
- Inspector layers 1–3 work.
- Costs: Kaggle.

Two decisions are made and won't be revisited in this doc: accounts from `docs/SETUP.md` are all
set up and working, and the brain simulator is **spiking**, built first — not rate-based. Rate-based
is still coming (M1.5's benchmark needs both), but spiking goes first because the calibration gate
means reproducing Shiu et al. 2024's actual published result, and that result is itself a spiking
model — going rate-based first would mean translating their spiking weights into a rate
approximation *before* calibration could even be attempted, which is an extra place to be wrong
before the one thing M1 has to get right is even checked.

## Why this order

Everything below is sequenced so that the expensive, hard-to-reverse step — trusting the brain —
happens as early as possible and as cheaply as possible, before a single line of sensory/motor
wiring or inspector code depends on it. If calibration fails, I want to find out on step 3, not
step 9.

---

## Step 0 — Wire real target movement (against the toy brain)

Found during review, not in the original scoping: `OdorSource.move_to()` already exists and is
unit-tested in isolation, but nothing actually uses it end to end. `Sim.__init__` reads
`scenario.targets[i].position` exactly once and bakes it into three separate places that never
look at it again — `TargetAssignment`'s geodesic distance field (`brains/targeting.py`), `agents.py`'s
capture-radius/feed logic (`sc.targets[own_target]`), and `viewer_export.py`'s marker position. Call
`move_to()` today and the smell moves while capture and routing and the viewer marker all stay put
at the old spot — three things that need to move together, only one of which does. Since M1's own
done-when line is "finds a *moved* target," this has to be real, working plumbing, not just an API
that exists.

This is independent of the real brain entirely, so it's built and tested against the toy brain
first, the same way every other piece of M0 plumbing was proven before the real connectome touched
it — cheap, fast feedback, fully covered by the existing CPU-only CI.

- Add a way to actually trigger a move during a run — either `Sim.move_target(index, new_xy)`
  callable from a script, or a scenario-level `[[target_moves]]` schedule (`at_s`, `target`, `to`),
  whichever fits `record_demo.py`'s existing CLI shape better.
- On a move: call the `OdorSource.move_to()`, rebuild the affected `TargetAssignment`'s geodesic
  field (targeting.py already recomputes this per-target when built; moving one just means
  redoing that one target's field, not every target's), and update whatever `agents.py` and
  `viewer_export.py` read so capture radius and the drawn marker both reflect the new position —
  not the scenario's original static config.
- New tests: an agent already walking toward the old position re-paths toward the new one; capture
  only succeeds at the new position, not the old one; the exported recording's marker position
  actually changes at the scheduled time.

Done when: a toy-brain fly, mid-run, visibly re-routes to a target that moved, capture only
triggers at the new spot, and it's covered by tests that run in the existing CI — before any of
Steps 1–6 below start.

**Status: done, 2026-09-26.** (Missing this line was itself a finding of the 2026-09-27 review pass
— Steps 1 and 2 both had a dated status line and this one didn't, even though `docs/JOURNAL.md`
confirms it landed before Step 1 started.)

## Step 1 — Pin the data

Nothing else can start on solid ground until this is done; it's also PLAN.md's own "still to
decide" items #3 and #7, both due by M1.

- **Pin the MaleCNS version.** `male-cns:v1.0` is already the dataset named in SETUP.md's C3
  check — confirm it's still the current tag on neuPrint, and write the exact tag down in
  `docs/JOURNAL.md` and in code (a constant, not a magic string scattered around).
- **Record hashes.** Once the bulk data is pulled (see below), hash the files and store the
  hashes alongside the pulled copy, so a silent dataset revision upstream is detectable later —
  PLAN.md's own risk table names this ("dataset gets revised under me", mitigation: "version
  pinned").
- **Bulk download, don't query per neuron.** neuPrint has a bulk-export path for the connectivity
  table (neuron list + synapse-weighted edge list); this is what gets pulled and hashed, not
  thousands of individual per-neuron API calls.
- **Confirm annotation coverage.** Before building anything that depends on a specific neuron
  type existing in the data, check that every type named in PLAN.md's sensory/motor tables
  (Or42b, Gr64f, Gr5a, Or47b, LC10, LC16, Johnston's organ, DNa02, DNa01, PFL2, DNp09, MDN, the
  giant-fibre pathway, PFL3, MBON32) is actually present and annotated in `male-cns:v1.0` —
  neuron *type*, not necessarily individual, since some of these are populations. Anything
  missing gets flagged and either substituted (nearest known equivalent) or the plan gets
  adjusted, in the open, before it silently becomes "that channel just doesn't do anything."
- **Check for the `flywireType` cross-reference column.** MaleCNS's own neuPrint annotation table
  is reported to carry a `flywireType` field alongside `type`/`instance`/`class` — i.e. Janelia
  may have already done the FlyWire↔MaleCNS cell-type matching that Step 3 needs, rather than it
  being a from-scratch task. Confirm this directly against neuPrint during this step (don't take
  a third party's word for it) and, if it's there, treat it as the mapping source of truth.
- **Where it lives.** Cloud only, per PLAN.md's own data rule — the pulled connectivity table and
  its hash file go into a Kaggle dataset (or another free object store), never onto the laptop,
  never into the git repo.

**Step 1 status: done, 2026-09-26.** Confirmed live against neuPrint and the downloaded
`body-annotations` table on Kaggle (see `flybrainflow/data_config.py` and `data/VERSION` for the
full detail, `docs/JOURNAL.md` for the narrative):
- `male-cns:v1.0` confirmed as the current, still-served tag.
- `flywireType` column confirmed present on the live annotation table.
- Every required type confirmed present **except** `Or42b`, `Gr64f`, `Gr5a`, `Or47b` — a real,
  confirmed gap (checked under `type`, `flywireType`, and `receptorType`, zero matches on all
  three; the antennal nerve itself is present in the volume, but MaleCNS never assigned these
  fibers receptor-gene-level identity). `LC10` is not actually missing — it's split across 7
  subtypes (`LC10a`–`LC10e` etc., 960 neurons) rather than one bare string.
- Johnston's organ (33 subtypes, 672 neurons) and the giant-fibre pathway (6 types, 38 neurons)
  both confirmed present.
- Bulk connectivity table pulled and hashed (3 of the ~11 bucket files — the ones Step 2 actually
  needs; see `data/README.md`).

This is the answer to the "what exactly gets substituted" open question further down this
document: **Or42b/Gr64f/Gr5a/Or47b**, specifically. Resolution deferred to Step 4 (sensory model) —
see `flybrainflow/data_config.py`'s `KNOWN_GAPS` comment for the three live options.

Done when: a pinned dataset tag, a hash file, and a written confirmation (in `docs/JOURNAL.md`)
that every named neuron type is present — or an explicit note on what's substituted for what's
missing.

## Step 2 — Spiking simulator scaffold

The actual simulator: takes a connectivity table (neurons + weighted, signed synapses) and steps
it forward in time, spike by spike, for one fly.

- **Sub-task 0 — done, 2026-09-26.** Verified the two pulled files actually join: every body-id in
  `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` (25,568,639 weighted
  edges; columns `body_pre`, `body_post`, `weight`, `type_pre`, `type_post`) is present in
  `body-annotations-male-cns-v1.0-minconf-0.5.feather` — 0% missing, clean join, no reconciliation
  step needed before Step 2's graph-building work starts. Also resolved, while in there: the
  211,577-vs-166,691 neuron-count question flagged during the review pass — 165,122 annotation rows
  are `status == "Traced"` (matches the published figure within ~1%), the other 46,455 are
  orphan/glia/unimportant/etc. bodies the raw table includes but Janelia's headline count doesn't.
  Full numbers in `flybrainflow/data_config.py` (`TRACED_STATUS`) and `data/README.md`.
  Real pulled file sizes (measured 2026-09-26, worth having on hand for the GPU-memory/reload risk
  in `docs/PLAN.md`'s risk table): `body-annotations-...feather` 14.5MB, `body-neurotransmitters-
  ...feather` 43.3MB, `connectome-weights-...-significant-only.feather` 502.2MB.
- **The brain-interface contract, decided now, not discovered mid-build.** `Sim` calls every brain
  through one fixed shape — `desired_velocities(ids, positions, radii, max_speed_mps,
  preferred_targets, *, personalities, odor_field, dt)`, `assigned_targets()`, `forget(ids)`,
  `move_target(index, new_xy)` — and `ConnectomeBrain` has to implement exactly that, the same as
  `Baseline`/`ToyBrain`. Two things that shape doesn't obviously cover, resolved here so Step 2 is
  built against a real contract instead of an assumption:
  - **Clock ratio.** `Sim` only ever passes one `dt` (the world tick) into `desired_velocities`;
    there's no second "brain clock" parameter anywhere, and the spiking simulator's own internal
    step size will not be 1:1 with the world's. Mechanism: `ConnectomeBrain` owns a private
    `self._brain_dt` (a constructor argument, tuned during Step 3's calibration gate — not decided
    yet, deliberately) and internally runs `round(dt / self._brain_dt)` spiking sub-steps inside
    one `desired_velocities` call before producing a single velocity output. `Sim`/`Population`
    never need to know this is happening — the seam stays exactly as clean as it is today. The
    cost is that one `desired_velocities` call's wall-clock time now depends on the clock ratio,
    which is a performance question for Step 3/M1.5's benchmark, not Step 2's — Step 2 just has to
    build the sub-stepping loop this way from the start rather than assuming one call is one step.
  - **`last_sense` / `Recorder._SENSE_WIDTH`.** Today this is an informal convention, not a
    documented interface: `ToyBrain` exposes `last_sense` as a small fixed-width array,
    `Recorder._SENSE_WIDTH = 6` is a hardcoded constant matching it, and `Baseline` doesn't expose
    `last_sense` at all. A real connectome brain's natural per-tick output is spike/activity state
    across ~166k–211k neurons — recording that in full, every tick, for every fly, is both the
    wrong design (Step 5's inspector layer 3 already handles "see the whole brain" by *recomputing*
    on demand from a recorded seed + inputs, not by recording it live) and not what `last_sense` is
    for. Decision: `last_sense` for `ConnectomeBrain` is a compact vector of exactly the values
    that actually drove that tick's behavior — the sensory-injection input values plus the
    motor-readout neurons' activity used to compute the steering command — not full per-neuron
    state. Its width is real but brain-specific, so `Recorder._SENSE_WIDTH` (currently a hardcoded
    module constant, `flybrainflow/recorder.py:38`) needs to become something the brain itself
    reports (e.g. a `sense_width` property every brain implements, `Recorder` reads at construction
    time) instead of a number someone has to remember to hand-edit when a new brain type shows up —
    make this change as part of Step 2, not as a Step 5 surprise.
- **Borrow code, not data.** Shiu et al.'s own published code
  (github.com/philshiu/Drosophila_brain_model, confirmed MIT-licensed) and the community
  embodied-fly project are both fair to read and reuse as *code* — their model structure, their
  spiking update rule, their parameter choices. Their FlyWire-derived *data* is not — that stays
  out of this repo and out of MaleCNS-side runs entirely, per PLAN.md's licensing rule ("This
  project doesn't get sold while anything derived from non-commercial data exists anywhere in the
  pipeline"). A smaller side-project port of the same model to both FlyWire v630 and MaleCNS
  v1.0 (also MIT) is useful as a second reference and as a feasibility sanity-check (numbers
  below) — treat it as a helpful indicator, not an authority, and verify anything load-bearing
  (the `flywireType` column, licence terms) directly rather than trusting its README.
- **Feasibility is comfortable, not a gate.** That side project reports FlyWire v630 (127,400
  neurons, 14.7M synapses) at ~665MB peak memory and MaleCNS v1.0 (166,700 neurons, 24.5M
  synapses) at ~434MB peak, with full-brain runtimes well under a second of compute per simulated
  second on ordinary hardware. Either connectome is a rounding error against a Kaggle T4's 16GB —
  there's headroom for many flies batched, not just one. No separate feasibility spike needed
  before starting this step.
- **Determinism is a design constraint, not an afterthought.** Floating-point atomic-add spike
  accumulation on GPU is not bit-reproducible run to run, even with a fixed seed — batched,
  parallel reductions can land in a different order each time. Build the update rule around
  integer (fixed-point) spike accumulation instead, set `torch.use_deterministic_algorithms(True)`,
  and avoid fused/compiled kernels (e.g. `torch.compile`) that reorder floating-point operations.
  This matters beyond tidiness: Step 5's inspector layer 3 depends on "same seed + same recorded
  inputs → same activity, always" being literally true, not just usually true.
- **One shared weight matrix, many flies.** The scaffold takes the connectivity table once and
  builds one sparse weight matrix; each simulated fly carries only its own spike/activity state,
  a seed, and its personality dials — the same pattern the M0 toy brain already uses (see
  `ToyBrain`'s own docstring: "one shared weight matrix for all flies"). Getting this right now,
  on one fly, is what makes running many flies later (M1.5, M2) a batching problem instead of a
  redesign.
- **No learning.** Weights are fixed from the connectome; nothing here trains anything.
- **Guard rails from day one.** Excitatory/inhibitory signs on the connectome are
  machine-predicted, not certain — PLAN.md is explicit that a few wrong signs across tens of
  millions of connections could make a brain run away (activity blows up) or go dead (activity
  collapses to zero), and a small test run wouldn't show it. So: activity clipping/bounding built
  into the update rule itself (not bolted on after), plus an explicit sanity check run on the
  full-size connectome specifically (not just a small test slice) before trusting any output from
  it. `ToyBrain`'s own bounded-activity test
  (`test_hidden_activity_never_exceeds_its_guard_rail_under_an_adversarial_drive`) is the shape to
  repeat here, adversarial inputs included.
- **Reproducible.** Same seed, same run — each fly's own `numpy.random.Generator` (or the PyTorch
  equivalent) seeded from `(scenario_seed, agent_id)`, matching the pattern already in place for
  the toy brain and the rest of M0.
- **Runs on GPU, batched.** PyTorch, built so that running N flies is N brains stacked into one
  batched tensor op, not a Python loop over N brains — this doesn't have to be *fast* yet (that's
  M1.5's job) but the shape has to be right from the start, since retrofitting batching into an
  already-written single-fly simulator is much more work than building it batched-from-one.

- **Two test tiers, not one.** GitHub Actions has no GPU, ever — the calibration gate and any
  full-connectome run are inherently manual, Kaggle-only, hand-logged work. But the simulator's
  *logic* (the update rule, the guard rails, the determinism check) shouldn't only get exercised
  that way: it gets its own fast, CPU-only automated tests against a small synthetic fake
  connectome (a few dozen neurons, hand-built weights with a known expected outcome), the same
  role `ToyBrain`'s test suite already plays, running in the same `tests.yml` CI as everything
  else. The full connectome is Kaggle-only and journal-logged, same as the calibration gate itself;
  the simulator code is CI-tested like everything else in this repo.
- **Code reaches Kaggle by commit hash, not by copy-paste.** Codespaces auto-syncs with git; a
  Kaggle notebook doesn't. Every Kaggle notebook cell that runs this simulator does
  `pip install git+https://github.com/NathanTaylorOps/Fly-Brain-Flow.git@<commit>` against a
  specific pinned commit, and that commit hash gets written down in `docs/JOURNAL.md` alongside
  whatever result the run produced — so "the gate passed" is a claim tied to an exact, reproducible
  version of the code, not something that quietly drifted between when it passed and when it's
  looked at again later.

Done when: the scaffold runs one fly's full connectome forward in time, on a GPU, without
crashing or blowing up, on made-up/neutral input — before it's asked to do anything sensible yet.

**Status: DONE, 2026-09-27.** Both halves of Step 2's verification are complete.
`flybrainflow/spiking/` (`connectivity.py`, `dynamics.py`, `simulator.py`) implements the LIF
update rule described above: integer/fixed-point accumulation via the float64-exact-integer trick
(see `dynamics.py`'s own docstring for why that's the actual determinism mechanism, not just
`torch.use_deterministic_algorithms`), guard rails on membrane potential, refractory handling, and
the "one shared weight matrix, many flies" batched-from-one shape.

Logic tier: 169/169 tests passing on a real machine (one real bug caught and fixed on the first
run — see `docs/JOURNAL.md`). `.github/workflows/tests.yml` updated to install the `brain` extra
(`torch`) so these run in CI going forward, not just locally.

Full-connectome GPU tier: `scripts/run_full_connectome_check.py` run for real on Kaggle against the
actual pulled MaleCNS data — 164,740 neurons (unique body-ids with at least one surviving edge in
the significant-only weight table — a graph-node count, not the same thing as the ~166.7k
published headline neuron count or the 165,122 `Traced`-status count; see `data_config.py`'s
`TRACED_STATUS` comment for how those relate), 25,568,639 edges, on an actual GPU, 200 spiking
sub-steps, no crash, activity guard rail held, 2.94s total (14.71ms/step). Zero spikes and zero
membrane activity in this run is expected, not a bug: the input was all-zero/neutral and the start
state was all-zero, so there was nothing to drive a spike. This check only proves the machinery
survives real scale on real hardware, per Step 2's own "done when" criterion — whether the brain
does anything sensible is Step 3's job, with real sensory input.

Run off the `main` branch at commit `e3861b4` (the repo needed to be flipped from Private to Public
on GitHub first — anonymous pip/curl/gcsfs requests from Kaggle all 404 against a private repo,
which looks identical to a wrong URL or a missing file; worth remembering if this ever needs
re-running from a fresh Kaggle session). Full console output and the `SUMMARY` block are logged in
`docs/JOURNAL.md`.

**One promise this section made and did not keep, caught by the 2026-09-27 review pass, not swept
under the rug:** the `last_sense`/`Recorder._SENSE_WIDTH` decision above says explicitly "make this
change as part of Step 2, not as a Step 5 surprise." It didn't happen — `flybrainflow/recorder.py:38`
still has `_SENSE_WIDTH = 6` as a hardcoded module constant, completely unchanged. On inspection this
turned out to be more than a rename: today's design assumes one global sense-width shared by every
cohort's brain (mixed cohorts get NaN-padded to that same width), and a real `ConnectomeBrain` will
almost certainly have a different width than `ToyBrain`'s 6 — so making this brain-reported rather
than hardcoded also means deciding how a recording holds two different brains' two different widths
at once, which the current NumPy-array-per-column format doesn't obviously support. That's a real
design question, not safe to rush through blind — moved to Step 3's pre-work list below rather than
silently deferred again.

## Step 3 — The calibration gate

This is the actual trust-building step, and it's the one PLAN.md is strictest about: "I reproduce
that result with my own build — on the same dataset they used — before I trust my build on
anything. Then I switch datasets and check again. One variable at a time, so if it breaks I know
which thing broke it."

**The concrete pass/fail bar** (decided now, before any run, specifically so the result can't get
eyeballed into a pass after the fact): stimulate sugar-sensing gustatory neurons and read out
motor neuron MN9 (controls proboscis extension) — this is Shiu et al.'s own primary validation
circuit. The gate is a **shuffled-connectome control**, not an absolute number match: at 100 Hz
stimulation, the real connectome should activate MN9 in close to 100% of runs, while a
degree-preserving *shuffled* version of the same connectome (same node degrees, randomised
edges) should activate it in only a small fraction (their own published result: ~1 in 100). This
was picked over trying to match their full 106-cell-type optogenetic screen because it's
self-contained — it doesn't need me to reproduce their entire external experimental dataset, just
the structural claim that the *real wiring specifically* (not just "a plausible amount of wiring")
is what produces the feeding response.

**Pre-work, found missing by the 2026-09-27 review pass — none of this exists yet, and the
protocol above cannot run without it. Listed here explicitly so it's owned and scoped before the
gate is attempted, not discovered mid-run:**

- **Sign resolution (excitatory/inhibitory).** The weights `load_connectivity` builds today are raw,
  unsigned synapse counts — there is currently no inhibition anywhere in the simulated network at
  all. Three places in this project disagreed about whose job this is (`connectivity.py`'s docstring
  said Step 4; this file's own Step 2 status implied Step 3; neither Step 3's nor Step 4's actual
  task list named it) — it belongs here, first, because a purely-excitatory ~165k-neuron recurrent
  network is a real runaway-activity risk on its own, independent of "a few wrong signs," and
  because the sugar-GRN → MN9 contrast this whole gate depends on is not a meaningful test without
  real inhibition in the circuit. `body-neurotransmitters-male-cns-v1.0.feather` was already pulled
  in Step 1 specifically for this. Concretely: map each neuron's predicted neurotransmitter to a
  sign convention (which transmitters are excitatory vs. inhibitory — this needs its own short
  literature check, not a guess), apply it per pre-synaptic neuron when building the weight matrix
  (every outgoing edge from an inhibitory neuron gets a negative weight), and write the convention
  down here once decided.
- **Degree-preserving shuffle.** The gate's control condition needs "the same connectome with edges
  randomised but node degrees preserved," and no such utility exists in this repo yet. Before
  building it, decide and write down: in-degree and out-degree preserved separately, or just total
  edge count? Is the weight distribution preserved (reshuffle which pairs share an edge, keep the
  multiset of weights) or resampled? How are the summed duplicate-pre/post pairs from
  `load_connectivity`'s coalescing handled — reshuffle before or after that aggregation? Get this
  decided and implemented as its own tested utility (small synthetic connectome first, same pattern
  as `flybrainflow/spiking/`'s own build) before it's needed for a real run.
- **Cell-type → matrix-index lookup.** `Connectivity` deliberately carries only `body_ids`, no type
  information (by design, so the spiking core stays connectome-source-agnostic) — but "stimulate
  sugar-GRNs, read out MN9" needs to go from a cell-type name to a set of matrix indices. This is
  glue code joining back to the annotation table (`body-annotations-male-cns-v1.0-minconf-0.5.feather`,
  already pulled), not a change to `Connectivity` itself — keep the separation; add a small lookup
  function that takes a loaded annotation dataframe and a type name/pattern and returns the matching
  `body_id`s to pass to `Connectivity.index_of()`.
- **`last_sense`/`Recorder._SENSE_WIDTH`**, carried over from Step 2 not actually being finished —
  see the note in Step 2's status section above for why this needs real design thought (a
  brain-reported width, and a decision on how a recording holds two different brains' two different
  widths at once), not just a rename. Only actually needed once `ConnectomeBrain` exists and gets
  recorded (Step 4), but the decision should be made before Step 4 starts building on top of it, so
  raising it here rather than letting it slide to another "Step 5 surprise."
- **The overflow guard's cost and correctness, found by the 2026-09-27 code review** (not urgent for
  a 200-step smoke test, but Step 3 plausibly needs thousands of steps to see steady-state firing
  rates, which is where this starts to matter): `dynamics._synaptic_input` recomputes
  `weight_matrix.values().abs().sum()` — a full-matrix reduction plus a GPU-to-CPU sync — on *every
  single tick*, even though the weight matrix never changes after construction. It also checks the
  wrong quantity relative to what its own docstring justifies: the safety argument is a per-row
  (per-postsynaptic-neuron) worst case, but the code sums absolute weight over the *entire* matrix,
  which is a much larger, unrelated number — safely under the ceiling for MaleCNS today by luck of
  scale, not because the check verifies the actual claim. Fix: precompute the true per-row max
  absolute weight sum once (e.g. as a field on `Connectivity`, computed in `load_connectivity` from
  the pre-torch NumPy arrays), and check it once at `SpikingSimulator` construction rather than
  every tick. Not done tonight because it touches the exact function `test_overflow_guard_trips_...`
  exercises directly — needs a matching test-file update and a real pytest run to confirm 169/169
  still holds, not a same-night blind edit.
- **The M1.5 feasibility claim isn't actually validated by tonight's real number yet.** Step 2's
  text asserts "feasibility is comfortable" based on a third-party FlyWire benchmark, not this
  project's own measurement. Tonight's real number (14.71ms per internal sub-step, one fly, zero
  activity) can't yet be converted to "seconds of compute per simulated second" because the clock
  ratio (`brain_dt`) is still undecided — and separately, `_synaptic_input`'s full-matrix
  int64→float64 conversion is a fixed per-tick cost independent of `n_flies`, so a single-fly
  measurement won't reveal how this actually scales once M1.5 batches many flies at once. Worth
  measuring for real, with the clock ratio decided, before M1.5's GPU-count benchmark is scheduled
  on the assumption that tonight's number already answers the feasibility question.

1. **Reproduce Shiu et al. on FlyWire.** Run the Step-2 scaffold on FlyWire data — the same
   dataset the published result used — with the sugar-GRN → MN9 protocol above, real connectome
   vs. shuffled control. FlyWire is signed into already (Codex, per SETUP.md C4) and is used here
   *only* for this one check; it never ships with the project and never touches MaleCNS-side code
   or data.
   - If this fails: the bug is in the simulator (Step 2), not the dataset, because the dataset is
     the one they validated on. Fix the scaffold, not the data.
2. **Switch to MaleCNS, re-check.** Same scaffold, same protocol, now on the pinned `male-cns:v1.0`
   connectome (Step 1), using the `flywireType` cross-reference (if confirmed in Step 1) to locate
   the equivalent sugar-GRN and MN9 cell types. Shiu et al.'s fitted constants (thresholds, weight
   scaling) were tuned for FlyWire's numbers specifically, so **matching FlyWire's absolute firing
   rates on MaleCNS is not the bar** — that's exactly why the gate is the shuffled-vs-real
   *contrast*, not an absolute number: a different reconstruction is expected to produce different
   rates, but the real wiring should still clearly out-activate its own shuffled control.
   - **Pass:** the contrast is present — real connectome clearly activates MN9 more than its
     shuffled control does — even if the absolute rates differ from FlyWire's. Different rates
     were already expected; a present contrast is the actual claim being tested.
   - **Fail:** the contrast is absent or weak — real and shuffled look similarly likely to
     activate MN9. This blocks Step 4 until root-caused (most likely candidates: a missing/
     mislabeled neuron type from Step 1's coverage check, or a wrong-sign connection at scale) —
     it is not something to route around by quietly reusing FlyWire-fitted constants on a
     MaleCNS-driven simulation.
3. **Write the gate result down**, pass or fail, in `docs/JOURNAL.md` — this is the actual
   go/no-go for the rest of M1. Everything past this point (Steps 4–6) assumes the gate passed on
   MaleCNS.

Done when: the calibration gate has a written, dated pass/fail entry in the journal, for both
datasets, with the one-variable-at-a-time trail showing which dataset (if either) it passed on.

## Step 4 — Wire senses and motors, on MaleCNS

Only once Step 3 has passed does the real connectome get connected to the M0 world it'll actually
navigate.

- **Sensory injection**, reusing M0's existing fields rather than rebuilding them:
  | Signal | Driven by | M0 code already there |
  |---|---|---|
  | Or42b (food smell) | odour concentration/gradient at the fly's position | `OdorField.sample/gradient(source=...)` |
  | Gr64f / Gr5a (food taste) | already handled — capture happens via `Population.feed`'s slot/radius logic, not a brain input | `agents.py` |
  | Or47b (mate smell) | stretch scope for M1 — not required for the done-when criteria | — |
  | LC10 (mate sight) | optional/stretch | — |
  | LC16 (other flies / walls) | nearby-agent and nearby-wall proximity, left/right split | `brains/repulsion.py` (`agent_repulsion`, `wall_repulsion`) |
  | Johnston's organ (wind) | local wind velocity | `world/airflow.py` (`AirflowField.velocity_at`) |
  | Leg touch | minimal — physics handles actual collision | `world/physics.py` |
  Deliberately left silent, per PLAN.md, as half of "no flying" enforcement: LC4, LC6, LPLC1,
  LPLC2 ("something's coming at me, jump" neurons).
- **Motor readout**, same "read descending neurons as turn rate + forward speed" shape the toy
  brain already established:
  - DNa02 (fast turn, left−right; fed by PFL3 compass + MBON32 "is this smell good") and DNa01
    (slow/sustained turn) combine into turn rate; PFL2 sets turning gain.
  - DNp09 → forward speed, including its documented "very high drive = stop" quirk — same curve
    shape as `ToyBrain._forward_speed_curve`, now read from the real neuron's activity instead of
    a hand-built formula.
  - MDN (backward) is a v1 cut, same as the toy brain — forward speed clamps at zero rather than
    going negative.
  - The giant-fibre (jump/take-off) pathway is read for the inspector display only, never acted
    on — the other half of "no flying."
- **Fits the existing brain interface**, not a new one: `desired_velocities(ids, positions, radii,
  max_speed_mps, personalities, odor_field, dt, preferred_targets)` is the shape `ToyBrain` and
  `Sim._steer()` already use (see `flybrainflow/sim.py`). The real brain implements that same
  shape — same non-holonomic heading-then-forward locomotion, same per-agent `numpy.random.
  Generator` seeding, same `.forget(ids)` cleanup on leave — so it slots into `Sim` with a
  one-line change (which brain class gets instantiated), not a rewrite of `Sim`, `agents.py`, or
  `physics.py`.
- **Personality is accepted, not used yet.** `desired_velocities(...)`'s signature requires a
  `personalities` argument — the real brain takes it and documents, explicitly, that it's a no-op
  for now. The real 20-row personality table is M2 scope (same as the toy brain's own docstring
  already says); M1 isn't the place to improvise a partial personality mechanism just because the
  interface happens to demand the parameter.
- **Clock ratio** (PLAN.md's "still to decide" #10, due by M1): the real spiking simulator runs on
  its own internal clock (brain milliseconds per tick), separate from the world's `dt`. Decide
  and document a default ratio for pedestrian-scale walking (the only mode M1 needs) once Step 3's
  calibration run gives a feel for how many brain-ms of settling a sensible motor decision
  actually needs — this is an empirical call, not a guess made in advance, so it's placed here
  rather than in Step 1.

Done when: one real-brain fly, dropped into an M0 scenario (a target moved partway through the
run, matching the done-when wording "finds a moved target"), walks a path to it that a person
looking at the viewer would call a search — better with wind on than with wind off, matching
PLAN.md's risk-table item "flies can't find the food" and its stated mitigation ("wind + real
upwind circuit... tested with and without at M1").

## Step 5 — Inspector layers 1–3

Layer 4 (recompute the whole 166k-neuron brain live) is explicitly not required until M3. M1 needs
layers 1–3:

- **Layer 1 — coloured path.** For every tick of the run, what was driving the fly at that moment
  (chasing food / chasing mate / avoiding someone / feeding), pre-stored per fly, budgeted around
  200 KB per fly-lifetime, must play back instantly with no recompute.
- **Layer 2 — region timeline.** A timeline of roughly 80 brain regions lighting up alongside that
  path, same pre-stored/instant requirement.
- **Layer 3 — full brain at an instant.** All 166k neurons, in 3D, at any scrubbed-to instant;
  recomputing on demand (seconds to a minute) is fine here, since it's one fly, one instant, not
  the whole run.
- **Layer 3 needs a small always-on backend — not just a static page.** The recompute itself is
  cheap (Step 2's feasibility numbers put a full-connectome run at well under a second of compute
  per simulated second), but a browser can't run it alone; something has to take recorded inputs +
  seed and return the neuron-activity snapshot on click. This is new infrastructure SETUP.md never
  covered — likely a small free-tier serverless endpoint (Render/Fly.io-style, not Kaggle, which
  isn't built for always-on hosting). Left as an open decision for the *start* of this step, not
  decided here, since it's new scope on top of everything else in this plan and deserves its own
  short conversation once Steps 1–4 are actually done.
- **What has to change to support this:**
  - `Recorder` already captures position, status, target and a 6-channel `sense` per tick
    (`flybrainflow/recorder.py`) — layers 1–2 need this extended: a "driving reason" label per
    tick (derived from which sensory/motor channel is dominant, or read directly off the brain if
    it exposes one) and enough of the per-tick sensory input to recompute the brain later, which
    the `sense` column mostly already is, just needs to carry the real brain's channel count
    instead of the toy's 6.
    layer 3's "recompute on demand" is exactly `Recorder`'s existing design promise — since every
    fly is deterministic (fixed seed, fixed weights, no learning), replaying its recorded inputs
    through the real brain reproduces its exact activity, so nothing about *all* 166k neurons'
    activity needs to be stored for every tick — just inputs + seed, which is already the shape
    `Recorder` captures.
  - **New viewer work**, not an extension of M0's existing 2D canvas playback (`viewer/
    index.html`) — PLAN.md is explicit the inspector is a Three.js web page, a genuinely different
    piece of code. M0's viewer stays as-is (it's already documented as a deliberate placeholder);
    the inspector is new, additive work, built once the recordings it reads (Step 4's runs, with
    the extended `Recorder` fields above) exist.

Done when: a recorded M1 run can be scrubbed in the inspector — path colour-coded by driving
reason, region timeline alongside it, and clicking any instant recomputes that one fly's full
brain in the browser within about a minute.

## Step 6 — Write it up

`docs/JOURNAL.md` gets the calibration gate result (already required by Step 3), plus a closing
M1 entry: what got built, what got cut (mate-seeking, backward locomotion — both already flagged
as v1 cuts in the toy brain's own docstring and now true of the real brain too), and the clock
ratio decided in Step 4. `README.md`'s milestone checklist and status line get updated the same
way M0's did.

---

## Where each step runs

| Step | Where | Why |
|---|---|---|
| 1 (data pull) | Kaggle notebook or Codespace (bulk download, no GPU needed) | neuPrint bulk export doesn't need a GPU |
| 2 (scaffold) | Codespace for the code, Kaggle for anything that needs a GPU to test | day-to-day logic doesn't need a GPU; running it on the full connectome does |
| 3 (calibration) | Kaggle (GPU) | both FlyWire and MaleCNS runs are full-size connectome runs |
| 4 (senses/motors) | Codespace for wiring, Kaggle to actually run a fly | same split as Step 2 |
| 5 (inspector) | Codespace (static web page, no GPU) | it's a browser page reading a recording file |

Recordings go to a Kaggle dataset or another free object store, same as Step 1's pulled data —
never into git. GPU sessions get closed when not in use, same rule as SETUP.md's E3.

## Risks specific to M1 (from PLAN.md's own risk table, carried forward here)

- **Calibration gate fails** (medium/high) — mitigation is already built into Step 3's
  one-variable-at-a-time sequencing: FlyWire first tells me whether it's the simulator; MaleCNS
  second tells me whether it's the dataset.
- **Brain runs away or goes dead, wrong connection signs** (medium/medium) — mitigation is Step
  2's guard rails plus a full-size sanity check before Step 3, not just a small test slice.
- **Flies can't find the food** (medium/high) — mitigation is the wind + upwind circuit, and
  Step 4's done-when criteria explicitly requires testing with and without wind, not just
  demonstrating the "with wind" case.
- **Neither brain type runs enough flies on cheap GPUs** — not M1's problem to solve (M1 is one
  fly at a time); noted here only because Step 2's batched-from-the-start scaffold is what keeps
  it from becoming a redesign at M1.5.
- **Dataset gets revised under me** (low/medium) — mitigation is Step 1's pinned tag and hashes.
- **Cloud bill creeps** (low/medium) — mitigation is the existing $0 Codespaces cap and Kaggle's
  free-tier GPU; nothing in this plan needs a paid tier.

## Issues to open (matching M0's pattern)

- `M1: wire real target movement end to end (toy brain, CI-tested)`
- `M1: pin MaleCNS version, hash, confirm neuron-type coverage`
- `M1: spiking simulator scaffold (batched, guard-railed, reproducible)`
- `M1: calibration gate — reproduce Shiu et al. on FlyWire`
- `M1: calibration gate — re-check on pinned MaleCNS`
- `M1: sensory injection (Or42b, LC16, Johnston's organ)`
- `M1: motor readout (DNa02/DNa01/PFL2 turn, DNp09 forward, clock ratio)`
- `M1: recorder — extend for inspector layers 1-2`
- `M1: inspector viewer (Three.js) — layers 1-3`
- `M1: journal + README milestone update`

## Open decisions still ahead (deliberately not resolved yet)

- **Step 5's backend hosting choice** for layer 3's on-demand recompute — needs its own short
  conversation once Steps 1–4 are done, not decided speculatively now.
- **What exactly gets substituted** if Step 1's coverage check finds a named neuron type missing
  from `male-cns:v1.0` — decided in the open, case by case, when (if) it actually happens.

## What this plan deliberately leaves for later, not for now

- Rate-based brain simulator and the M1.5 GPU-count benchmark comparing it to spiking — M1.5, not
  M1.
- The real 20-row personality table — M2, per the toy brain's own docstring.
- Mate-seeking (Or47b/LC10), dedicated stop neurons, backward locomotion (MDN) — all already
  named as stretch/cut items; nothing in M1's done-when criteria needs them.
- Inspector layer 4 (live full-brain recompute at scale) — M3.
- Car-scale mode — not needed until whatever milestone actually uses it.

---

Next: Step 3 — the calibration gate. Steps 0–2 are done (see their status lines above); before
Step 3 itself is attempted, its pre-work list needs to actually close out first — see Step 3's own
section for what that is.
