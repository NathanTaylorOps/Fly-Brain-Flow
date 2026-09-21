#!/usr/bin/env python3
"""Run a scenario for real and export it for `viewer/index.html` -- the fastest way to actually
see the plumbing built so far do something, rather than reading about it.

    python scripts/record_demo.py
    python scripts/record_demo.py scenarios/corridor_bidirectional.toml --duration 60
    python scripts/record_demo.py --scenario scenarios/corridor_bidirectional.toml --duration 60
    python -m http.server -d viewer 8000    # fetch() needs http(s), not file://
    # open http://localhost:8000/?data=run.json  (or use the file picker on the page directly)

`--duration` defaults to a shortened 60 simulated seconds rather than the scenario's own
`duration_s` (300, for `corridor_bidirectional.toml`) -- long enough to see the crowd actually
build up and flow, short enough that this stays a quick thing to run and doesn't produce a huge
recording file for what's still just an M0 sanity check, not a real M4 result run.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from flybrainflow.recorder import Recorder
from flybrainflow.scenario import Scenario
from flybrainflow.sim import Sim
from flybrainflow.viewer_export import export_for_viewer

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Real usability gap, fixed: this used to be `--scenario`-only, so a plain `python
    # scripts/record_demo.py scenarios/x.toml` -- the first, most natural thing to try -- failed
    # with an argparse "unrecognized arguments" error instead of just working. `scenario_positional`
    # (`nargs="?"`) accepts that same natural form without breaking the existing `--scenario` form
    # anyone's scripts or muscle memory already rely on; `_resolve_scenario` below picks whichever
    # one was actually given, preferring the explicit `--scenario` flag if someone passes both.
    parser.add_argument("scenario_positional", nargs="?", default=None, metavar="SCENARIO", help="path to a scenario TOML file (same as --scenario)")
    parser.add_argument("--scenario", default=None, help=f"default: {REPO_ROOT / 'scenarios' / 'corridor_bidirectional.toml'}")
    parser.add_argument("--duration", type=float, default=60.0, help="simulated seconds to run (default: 60, must be > 0)")
    parser.add_argument("--dt", type=float, default=0.1, help="seconds per tick (default: 0.1, must be > 0)")
    parser.add_argument("--out", default=str(REPO_ROOT / "viewer" / "run.json"))
    parser.add_argument("--seed", type=int, default=None, help="override the scenario's own seed")
    args = parser.parse_args()

    # Real bug, fixed: `--dt 0` (or a negative --duration/--dt) used to reach `args.duration /
    # args.dt` below and crash with a raw ZeroDivisionError (or silently run zero/negative ticks) --
    # no help at all for a non-programmer working from a terminal. A clear, named error is what
    # every other user-facing input in this project (scenario.py's own validation) already does.
    if args.duration <= 0:
        parser.error(f"--duration must be > 0, got {args.duration}")
    if args.dt <= 0:
        parser.error(f"--dt must be > 0, got {args.dt}")

    scenario_path = args.scenario or args.scenario_positional or str(REPO_ROOT / "scenarios" / "corridor_bidirectional.toml")
    sc = Scenario.from_toml(scenario_path)
    if args.seed is not None:
        sc = dataclasses.replace(sc, meta=dataclasses.replace(sc.meta, seed=args.seed))

    sim = Sim.from_scenario(sc)
    rec = Recorder(sim)
    n_ticks = int(round(args.duration / args.dt))
    for _ in range(n_ticks):
        sim.tick(args.dt)
        rec.capture()

    export_for_viewer(rec.to_arrays(), sim.map, sc, args.out)

    counts = {tag: len(pop.agents) for tag, pop in sim.cohorts.items()}
    print(f"Ran '{sc.meta.name}' for {args.duration:.0f}s ({n_ticks} ticks).")
    print(f"Recorded {len(rec)} rows -> {args.out}")
    print(f"Still walking/feeding at the end: {counts}")
    print()
    print("To watch it:")
    print(f"  python -m http.server -d {Path(args.out).parent} 8000")
    print(f"  open http://localhost:8000/?data={Path(args.out).name}")


if __name__ == "__main__":
    main()
