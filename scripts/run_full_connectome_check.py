#!/usr/bin/env python3
"""M1 Step 2's actual "done when" check: load the real, full-size MaleCNS connectome (pulled by
scripts/pin_dataset.py) and run flybrainflow.spiking's scaffold forward in time on it, on a GPU,
on made-up/neutral input, and confirm it does not crash or its activity blow up.

Kaggle-only, same as pin_dataset.py and for the same reason: the real connectivity table lives in
a Kaggle working directory (or wherever pin_dataset.py's --out pointed), never on the laptop, never
in this repo, and this check needs a real GPU to actually exercise the "runs on GPU" requirement
(a CPU-only run proves the logic works -- the CI-tier tests in tests/test_spiking_*.py already do
that against small synthetic connectomes -- but not that it survives real GPU batching at real
scale, which is the thing docs/M1_PLAN.md's Step 2 actually asks this specific check to prove).

Usage (in a Kaggle notebook, after pin_dataset.py has already pulled the data):

    !pip install torch  # if not already present in the Kaggle image
    !pip install git+https://github.com/NathanTaylorOps/Fly-Brain-Flow.git@<commit>
    !python run_full_connectome_check.py --data-dir /kaggle/working/malecns_v1_0

Per docs/M1_PLAN.md's Step 2 ("Code reaches Kaggle by commit hash, not by copy-paste"), pin the
<commit> above to an actual commit hash and write it down in docs/JOURNAL.md next to whatever this
prints -- "the scaffold survived a full-connectome run" is a claim tied to one exact version of the
code, not something that can drift silently between when it passed and when it's checked again.

Prints a plain-text SUMMARY block at the end -- paste that back, same as pin_dataset.py's own.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="directory pin_dataset.py downloaded the connectivity files into")
    parser.add_argument("--n-steps", type=int, default=200, help="how many internal spiking sub-steps to run (default: 200)")
    parser.add_argument(
        "--weights-file",
        default="connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
        help="filename (inside --data-dir) of the weighted-edge table",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        parser.error("pandas isn't installed -- run: pip install pandas")
    try:
        import torch
    except ImportError:
        parser.error("torch isn't installed -- run: pip install torch")

    from flybrainflow.spiking import LIFParams, SpikingSimulator, load_connectivity

    data_dir = Path(args.data_dir)
    weights_path = data_dir / args.weights_file
    if not weights_path.exists():
        parser.error(
            f"{weights_path} does not exist -- did scripts/pin_dataset.py already run against "
            f"--out {data_dir}? (check the filename too -- pass --weights-file if it's named "
            "differently than the default)"
        )

    print(f"Loading {weights_path} ...")
    t0 = time.time()
    weights_df = pd.read_feather(weights_path)
    print(f"  {len(weights_df):,} edges, columns: {list(weights_df.columns)} ({time.time() - t0:.1f}s)")

    print("\nBuilding the sparse weight matrix ...")
    t0 = time.time()
    # No sign resolution here -- see flybrainflow/spiking/connectivity.py's own docstring for why
    # that's deliberately out of scope for this scaffold-level check. Raw (positive, unsigned)
    # synapse-count weights are used as-is; this checks "does the machinery survive real scale",
    # not "is the resulting activity biologically meaningful" (that's Step 3's calibration gate).
    connectivity = load_connectivity(
        body_pre=weights_df["body_pre"].to_numpy(),
        body_post=weights_df["body_post"].to_numpy(),
        weight=weights_df["weight"].to_numpy(),
    )
    print(f"  {connectivity.n_neurons:,} neurons, weight matrix built ({time.time() - t0:.1f}s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device} ({'GPU found' if device == 'cuda' else 'NO GPU FOUND -- this run will not prove the GPU requirement, only the logic'})")
    if device == "cuda":
        connectivity = connectivity.to(device)  # see Connectivity.to() -- returns a new instance, doesn't mutate in place

    params = LIFParams()  # placeholder defaults -- see LIFParams' own docstring; Step 3 calibrates these for real
    sim = SpikingSimulator(connectivity, params, n_flies=1)  # follows connectivity's device automatically, see SpikingSimulator.__init__

    print(f"\nRunning {args.n_steps} internal spiking sub-steps on made-up/neutral (all-zero) input ...")
    t0 = time.time()
    zero_input = torch.zeros((1, connectivity.n_neurons), dtype=torch.int64, device=device)
    crashed = False
    max_abs_membrane = 0
    total_spikes = 0
    steps_completed = 0
    try:
        for i in range(args.n_steps):
            spikes = sim.step(zero_input)
            total_spikes += int(spikes.sum().item())
            max_abs_membrane = max(max_abs_membrane, int(sim.state.membrane.abs().max().item()))
            steps_completed = i + 1
    except Exception as e:
        crashed = True
        crash_message = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    # Report max_abs_membrane/total_spikes even on a crash -- found in the 2026-09-27 review pass:
    # gating these behind "if not crashed" discards exactly the diagnostic that would tell whether
    # a crash was an environment/library problem or the scaffold's own guard rails failing first
    # (activity already blowing past the guard rail right before the exception). steps_completed
    # (which step it died on) is reported for the same reason.
    print("\n" + "=" * 60)
    print("SUMMARY (paste this whole block back)")
    print("=" * 60)
    print(f"device: {device}")
    print(f"neurons: {connectivity.n_neurons:,}")
    print(f"edges: {len(weights_df):,}")
    print(f"steps run: {args.n_steps}")
    print(f"crashed: {crashed}" + (f" ({crash_message})" if crashed else ""))
    if crashed:
        print(f"steps completed before crash: {steps_completed} / {args.n_steps}")
    print(f"activity stayed within guard rail [{params.activity_min}, {params.activity_max}]: {max_abs_membrane <= params.activity_max}"
          + (" (partial, before crash)" if crashed else ""))
    print(f"max |membrane| observed: {max_abs_membrane}" + (" (partial, before crash)" if crashed else ""))
    print(f"total spikes over the run: {total_spikes:,}" + (" (partial, before crash)" if crashed else ""))
    print(f"elapsed: {elapsed:.2f}s ({elapsed / max(steps_completed, 1) * 1000:.2f}ms/step)")


if __name__ == "__main__":
    main()
