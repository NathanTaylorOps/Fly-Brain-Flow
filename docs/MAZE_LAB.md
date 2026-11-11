# Maze Lab

Run `python scripts/live_demo.py --seed 7` and open http://127.0.0.1:8765/.
The behavior mode needs only the base dependencies; there is no connectome download.
Use `--data-dir /path/to/public/data` to make the existing real-connectome mode
available in the controller selector. The graph loads only when selected.

## Controls

- Run/pause and 1×, 2× or 4× playback. This changes simulated-time advancement,
  not the model's wind speed.
- Wind compass or direction slider: direction the air travels toward;
  0° east, 90° north, 180° west, 270° south.
- Fan speed: 0–3 m/s. At zero, odor still diffuses but does not advect.
- Click an open passage to move food. Previously released odor dissipates
  naturally while new odor starts at the new position.
- Toggle odor, wind traces and the fly's actual travel trail.
- Disable smell sensing to compare exploration without odor input.
- Reset returns the selected controller to its seeded initial state.
- Experiment details exposes seed, model limitations and observation export.

## What is simulated

The 16×12 model-unit maze has three alternating gates and dead-end branches.
Airflow uses a potential-flow approximation with impermeable internal barriers
and ambient openings at the outer boundary. A positive finite-volume
advection/diffusion solver carries odor through open grid faces. The controller,
concentration view and animated particles sample the same wind field. Particles
illustrate transport and are not individual physical odor molecules. They cannot
pass through walls; their animation advances with simulated time.

The initial plume is prewarmed for 60 simulated seconds, taking a fraction of a
second on this machine. That odor age is shown separately from the fly's clock.
This is not turbulent CFD or a calibrated physical fruit-fly arena. Scale,
emission, diffusion and motor animation are illustrative.

## Controller modes

**Behavior model:** hand-written local odor search, upwind preference, exploration
memory, obstacle sensing, turning/casting and sniffing pauses. It never reads the
food's coordinates or a shortest-path route. Food capture triggers feeding.
Body, legs and wings are an illustrative animation of the returned movement.

**Real connectome:** the previously implemented measured MaleCNS graph and
engineered output decoder. This mode retains direct stimulation/silencing controls.
Natural food seeking remains unvalidated. The mode selector and page notice must
keep this distinction explicit; behavioral-model results do not validate neural control.

## Measured behavior

The final app-default trials (seeds 3, 7, 42) reached feeding in 37.55, 37.60 and
57.25 simulated seconds. A wider wind sweep reached food in 10 of 12 trials
within 180 seconds; the remaining two failed. Relocating food and reversing the
wind succeeded in 54.1 seconds in the tested run. No sampled motion segment
crossed a wall in the final evidence runs. These are model results, not measurements
of real fruit-fly behavior or guarantees for every wind setting.

Run `pytest tests/test_forager.py tests/test_live_maze.py tests/test_live_demo.py`.
The tests check food reaching, moved-target recovery, actual app setup, gate
crossings, mass conservation, odor blocked by sealed walls, wind reversal and API
controls. Raw measurement evidence is saved in the delivered
`foraging-validation.json`.
