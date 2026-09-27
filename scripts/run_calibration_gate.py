#!/usr/bin/env python3
"""M1 Step 3's actual "done when" check -- the calibration gate: stimulate a sensory cell type,
read out a motor cell type, and confirm the REAL connectome's readout response differs meaningfully
from a degree-preserving SHUFFLED control's -- the thing docs/M1_PLAN.md's Step 3 asks for, built on
top of the three pre-work pieces added 2026-09-27 (flybrainflow.spiking.neurotransmitters/shuffle/
celltypes) after that day's review pass found the plan's own protocol assumed machinery that didn't
exist yet.

Kaggle-only, same as pin_dataset.py and run_full_connectome_check.py, and for the same reasons
(torch, the real ~25.5M-edge connectivity table, the ~164,740-neuron scale).

TWO REAL GAPS THIS SCRIPT DELIBERATELY DOES NOT PAPER OVER (found 2026-09-27, see docs/JOURNAL.md
and docs/M1_PLAN.md's Step 3 pre-work section):

  1. Step 3's own protocol text says "stimulate sugar-GRNs, read out MN9" -- but Gr64f and Gr5a (the
     exact sugar-GRN type strings named) are already confirmed ABSENT from male-cns:v1.0 under those
     names (see flybrainflow.data_config.KNOWN_GAPS: they're FlyWire/hemibrain genetic-driver names,
     not maleCNS's own nomenclature), and MN9's real MaleCNS type string had never been confirmed
     anywhere in this codebase. Hardcoding either as if already-confirmed-working would produce a
     confusing `ValueError` at best (see celltypes.body_ids_for_type) or, worse, a falsely-passing
     comparison if a typo happened to match zero-but-not-erroring rows some other way. So: this
     script REFUSES to guess by default. Real, sourced candidates were found by research (not
     assumed) -- see flybrainflow.data_config's own CALIBRATION_GATE_* constants and comment for the
     full citation trail (a maleCNS-native taste-connectome paper naming LB3b/LB3c as the
     sweet-sensing GRNs and MN9 as the readout neuron, corroborated for MN9 by an independent
     third-party reimplementation's actual working code) -- but these are still UNCONFIRMED against
     the actual pulled annotation table (no live data access from where this was researched). Run
     with no --stimulus-type/--readout-type and it prints those sourced candidates plus broader
     DISCOVERY searches (every type string matching sugar/taste/motor-neuron patterns) so you can
     confirm or correct them against what's actually in the data, the same "print real data before
     assuming schema" discipline pin_dataset.py already uses for the annotation table's columns.
     Only once you've confirmed the real strings from that output do you pass them explicitly and
     the actual stimulate-and-readout protocol runs.

  2. The real column name(s) of body-neurotransmitters-male-cns-v1.0.feather have never been
     inspected live in this codebase either (see neurotransmitters.py's own docstring). This script
     prints that file's real columns first and tries a short list of plausible body-id/transmitter
     column spellings, failing loudly with the real column list if none match, rather than assuming.

Usage (in a Kaggle notebook, after pin_dataset.py has already pulled the data):

    !pip install torch  # if not already present in the Kaggle image
    !pip install git+https://github.com/NathanTaylorOps/Fly-Brain-Flow.git@<commit>

    # Step A -- discovery only (no --stimulus-type/--readout-type): prints sourced candidates
    # (LB3b/LB3c for stimulus, MN9 for readout -- see data_config.py's own comment for the citation
    # trail) plus broader real candidate type strings from the live table. Does NOT run the protocol.
    !python run_calibration_gate.py --data-dir /kaggle/working/malecns_v1_0

    # Step B -- if the sourced candidates check out in Step A's output, this is the most likely
    # first real run to try:
    !python run_calibration_gate.py --data-dir /kaggle/working/malecns_v1_0 \\
        --stimulus-type "LB3[bc]" --stimulus-regex --readout-type "MN9"

    # Step B' -- if Step A found something different instead, use whatever it actually found:
    !python run_calibration_gate.py --data-dir /kaggle/working/malecns_v1_0 \\
        --stimulus-type "<real type or regex>" --readout-type "<real type or regex>"

Per docs/M1_PLAN.md's Step 2/3 discipline, pin <commit> to an actual commit hash and write down
whatever this prints in docs/JOURNAL.md next to it. Prints a plain-text SUMMARY block at the end --
paste that back, same as the other two Kaggle-only scripts.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flybrainflow.data_config import (  # noqa: E402
    CALIBRATION_GATE_READOUT_TYPE_CANDIDATE,
    CALIBRATION_GATE_STIMULUS_TYPE_PATTERN_CANDIDATE,
    KNOWN_GAPS,
)

# Broad, deliberately loose search patterns used ONLY for discovery mode -- these are NOT assumed to
# be the real type strings, they're just plausible substrings/prefixes to narrow ~166k neurons'
# worth of `type` values down to a short, eyeballable candidate list, IN ADDITION to the specific,
# sourced candidates from data_config.py (see that module's own comment for the citation trail) that
# are checked first and separately below. "sugar"/"taste"/"gustatory" won't literally appear in a
# `type` string (MaleCNS types are short codes, not English words) -- these broader patterns are
# aimed at the KIND of code sugar-GRNs and MN9 tend to use elsewhere in this project's own
# already-confirmed data (Gr-prefixed for gustatory receptors, LB-prefixed for labellar bristle
# GRNs, MN-prefixed + digits for numbered motor neurons), not a guarantee either role is spelled
# this way in MaleCNS.
_DISCOVERY_PATTERNS = {
    "gustatory-receptor-like (Gr*)": r"Gr\d",
    "sugar/carbohydrate-receptor-like (Gr43/64/61 family)": r"Gr(?:43|64|61)",
    "labellar-bristle-GRN-like (LB*, per Tastekin et al.)": r"^LB\d",
    "numbered motor neuron (MN*, any digits)": r"^MN\d",
    "'9'-numbered motor neuron specifically": r"^MN.*9",
}

_NT_BODY_ID_CANDIDATES = ("bodyId", "body_id", "bodyid")
_NT_TRANSMITTER_CANDIDATES = (
    "predicted_nt", "predictedNt", "nt_type", "ntType", "transmitter",
    "predicted_transmitter", "neurotransmitter", "nt",
)


def _find_column(df, candidates: tuple[str, ...], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"could not find a {what} column in body-neurotransmitters-male-cns-v1.0.feather (tried "
        f"{list(candidates)}) -- columns actually present: {list(df.columns)} -- add the real name "
        "to the candidate list in this script (or pass it via a new --nt-*-column flag) once you've "
        "confirmed it from the printed column list above"
    )


def _discover_candidates(annotations, type_column: str) -> None:
    print("\n" + "=" * 60)
    print("DISCOVERY MODE -- no --stimulus-type/--readout-type given")
    print("=" * 60)
    print(
        f"Step 3's protocol names Gr64f/Gr5a (sugar-GRNs) and MN9 (motor readout) -- Gr64f/Gr5a are "
        f"already confirmed ABSENT from male-cns:v1.0 under those exact names (see "
        f"data_config.KNOWN_GAPS = {list(KNOWN_GAPS)}; those are FlyWire/hemibrain genetic-driver "
        "names, not maleCNS's own nomenclature). This script will not guess blind, but real research "
        "(2026-09-27, see data_config.py's own CALIBRATION_GATE_* comment for the full citation "
        "trail) found sourced candidates specific to maleCNS -- checked first, below, separately "
        "from the broader pattern searches."
    )
    types = annotations[type_column].dropna().astype(str)

    print(f"\n  SOURCED CANDIDATE -- readout ({CALIBRATION_GATE_READOUT_TYPE_CANDIDATE!r}, from an "
          "independent third-party reimplementation's working code, corroborated against maleCNS):")
    readout_matches = sorted(types[types == CALIBRATION_GATE_READOUT_TYPE_CANDIDATE].unique().tolist())
    print(f"    exact match count: {int((types == CALIBRATION_GATE_READOUT_TYPE_CANDIDATE).sum())}"
          + (f"  (found: {readout_matches})" if readout_matches else "  -- NOT FOUND under this exact string"))

    print(f"\n  SOURCED CANDIDATE -- stimulus (pattern {CALIBRATION_GATE_STIMULUS_TYPE_PATTERN_CANDIDATE!r}, "
          "from Ganguly/Tastekin et al.'s maleCNS-native taste-connectome paper -- UNCONFIRMED, see "
          "data_config.py's comment for why this one specifically needs a live check):")
    stim_matches = sorted(types[types.str.match(CALIBRATION_GATE_STIMULUS_TYPE_PATTERN_CANDIDATE, na=False)].unique().tolist())
    print(f"    matches: {stim_matches}" if stim_matches else "    -- NOT FOUND under this pattern")
    for label, pattern in _DISCOVERY_PATTERNS.items():
        matches = sorted(types[types.str.contains(pattern, regex=True, na=False)].unique().tolist())
        print(f"\n  {label}  (pattern: {pattern!r}):")
        if matches:
            for m in matches[:40]:
                print(f"    {m}")
            if len(matches) > 40:
                print(f"    ... and {len(matches) - 40} more")
        else:
            print("    (no matches)")
    print(
        "\nIf the sourced candidates above check out: re-run with "
        f"--readout-type {CALIBRATION_GATE_READOUT_TYPE_CANDIDATE!r} --stimulus-type "
        f"{CALIBRATION_GATE_STIMULUS_TYPE_PATTERN_CANDIDATE!r} --stimulus-regex to actually run the "
        "protocol. If neither the sourced candidates nor anything below looks like a real sugar-GRN "
        "or MN9 equivalent, the honest options are the same three data_config.py's own KNOWN_GAPS "
        "comment already lays out for Step 4's sensory side: cross-reference FlyWire/hemibrain by "
        "identity, fall back to a coarser proxy population, or treat the input/readout as an unnamed "
        "population -- or go back to the Tastekin et al. paper itself for a supplementary body-ID "
        "table, if the peer-reviewed version publishes one. That's a real decision for Nathan, not "
        "something this script should pick on its own."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="directory pin_dataset.py downloaded the connectivity files into")
    parser.add_argument("--annotations-file", default="body-annotations-male-cns-v1.0-minconf-0.5.feather")
    parser.add_argument("--weights-file", default="connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather")
    parser.add_argument("--nt-file", default="body-neurotransmitters-male-cns-v1.0.feather")
    parser.add_argument("--stimulus-type", default=None, help="real, CONFIRMED type string or regex for the sensory population to stimulate (see discovery mode if you don't have this yet)")
    parser.add_argument("--readout-type", default=None, help="real, CONFIRMED type string or regex for the motor population to read out (see discovery mode if you don't have this yet)")
    parser.add_argument("--stimulus-regex", action="store_true", help="treat --stimulus-type as a regex, not an exact match")
    parser.add_argument("--readout-regex", action="store_true", help="treat --readout-type as a regex, not an exact match")
    parser.add_argument("--n-trials", type=int, default=20, help="how many independent stimulate-and-read trials to run per condition (default: 20)")
    parser.add_argument("--n-steps", type=int, default=50, help="internal spiking sub-steps per trial (default: 50)")
    parser.add_argument("--stimulus-strength", type=int, default=50, help="external input value injected at each stimulus neuron, every step of the trial (default: 50 -- comfortably above the LIF threshold; see flybrainflow.spiking.dynamics.LIFParams defaults)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        parser.error("pandas isn't installed -- run: pip install pandas")

    data_dir = Path(args.data_dir)
    ann_path = data_dir / args.annotations_file
    weights_path = data_dir / args.weights_file
    nt_path = data_dir / args.nt_file
    for p in (ann_path, weights_path, nt_path):
        if not p.exists():
            parser.error(f"{p} does not exist -- did scripts/pin_dataset.py already run against --out {data_dir}?")

    print(f"Loading {ann_path} ...")
    annotations = pd.read_feather(ann_path)
    print(f"  {len(annotations):,} rows, columns: {list(annotations.columns)}")

    print(f"\nLoading {weights_path} ...")
    weights_df = pd.read_feather(weights_path)
    print(f"  {len(weights_df):,} edges, columns: {list(weights_df.columns)}")

    print(f"\nLoading {nt_path} ...")
    nt_df = pd.read_feather(nt_path)
    print(f"  {len(nt_df):,} rows, columns: {list(nt_df.columns)}")
    nt_id_col = _find_column(nt_df, _NT_BODY_ID_CANDIDATES, "body-id")
    nt_col = _find_column(nt_df, _NT_TRANSMITTER_CANDIDATES, "predicted-transmitter")
    print(f"  using {nt_id_col!r} as body-id, {nt_col!r} as predicted transmitter")

    if args.stimulus_type is None or args.readout_type is None:
        _discover_candidates(annotations, type_column="type")
        print(
            "\nStopping here (discovery mode) -- no --stimulus-type/--readout-type given. Nothing "
            "below this line ran; nothing was stimulated, nothing was read out, no PASS/FAIL was "
            "computed. Re-run with both flags set once you've confirmed the real type strings above."
        )
        return

    # -- sign resolution -----------------------------------------------------
    from flybrainflow.spiking import apply_signs, body_ids_for_type, resolve_signs, shuffle_connectivity

    print("\nResolving excitatory/inhibitory signs (Shiu et al. convention) ...")
    body_ids = nt_df[nt_id_col].to_numpy()
    transmitters = nt_df[nt_col].to_numpy()
    # unknown_sign=1 (excitatory), matching Shiu et al.'s own "everything except GABA/glutamate is
    # excitatory" rule applied to anything this map doesn't recognize -- NOT silently assumed: see
    # resolve_signs' own docstring for why this default is a real, named choice, not an oversight.
    signs = resolve_signs(body_ids, transmitters, unknown_sign=1)
    print(f"  resolved signs for {len(signs):,} neurons (of {len(nt_df):,} rows in the nt table)")

    body_pre = weights_df["body_pre"].to_numpy()
    body_post = weights_df["body_post"].to_numpy()
    weight = weights_df["weight"].to_numpy()
    n_no_sign = int((~pd.Series(body_pre).isin(signs.keys())).sum())
    if n_no_sign:
        print(
            f"  NOTE: {n_no_sign:,} / {len(body_pre):,} edges have a body_pre with no neurotransmitter "
            "prediction row at all -- defaulting those to excitatory (default_sign=1) below. This is "
            "expected (funkelab/synister_malecns' own documented class-imbalance/confidence-filtering "
            "caveats -- see docs/JOURNAL.md's 2026-09-27 research entry), not a bug, but the count is "
            "printed so it can be sanity-checked against what those caveats would predict."
        )
    signed_weight = apply_signs(body_pre, weight, signs, default_sign=1)

    # -- real vs. shuffled connectivity ---------------------------------------
    from flybrainflow.spiking import LIFParams, SpikingSimulator, load_connectivity
    import torch

    print("\nBuilding the REAL signed connectivity ...")
    t0 = time.time()
    real_connectivity = load_connectivity(body_pre=body_pre, body_post=body_post, weight=signed_weight)
    print(f"  {real_connectivity.n_neurons:,} neurons ({time.time() - t0:.1f}s)")

    print("\nBuilding the SHUFFLED (degree-preserving) control connectivity ...")
    print(
        "  NOTE: shuffle_connectivity's own module docstring flags this as correctness-first, not "
        "yet benchmarked at real ~25.5M-edge scale -- if this step is unexpectedly slow, that's the "
        "known, already-documented risk, not a new bug (see docs/M1_PLAN.md's Step 3 pre-work note)."
    )
    rng = __import__("numpy").random.default_rng(args.seed)
    t0 = time.time()
    shuf_pre, shuf_post, shuf_weight = shuffle_connectivity(body_pre, body_post, signed_weight, rng=rng)
    shuffled_connectivity = load_connectivity(body_pre=shuf_pre, body_post=shuf_post, weight=shuf_weight)
    print(f"  done ({time.time() - t0:.1f}s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        real_connectivity = real_connectivity.to(device)
        shuffled_connectivity = shuffled_connectivity.to(device)

    # -- cell-type lookup (only reached once both --stimulus-type/--readout-type are confirmed real) --
    print(f"\nLooking up stimulus population (type={args.stimulus_type!r}, regex={args.stimulus_regex}) ...")
    stimulus_ids = body_ids_for_type(annotations, args.stimulus_type, regex=args.stimulus_regex)
    print(f"  {len(stimulus_ids)} neuron(s)")
    print(f"Looking up readout population (type={args.readout_type!r}, regex={args.readout_regex}) ...")
    readout_ids = body_ids_for_type(annotations, args.readout_type, regex=args.readout_regex)
    print(f"  {len(readout_ids)} neuron(s)")

    # -- the actual stimulate-and-read-out protocol ---------------------------
    def _run_condition(connectivity, label: str) -> float:
        """Runs args.n_trials independent trials on `connectivity`, each: reset state, drive
        `stimulus_ids` with a constant strong input for the whole trial, run n_steps, and record
        whether ANY readout neuron spiked at ANY point in the trial. Returns the fraction of trials
        (across n_trials) with at least one readout spike -- the "activation rate" Step 3's protocol
        compares between the real and shuffled conditions."""
        id_to_idx = {int(b): i for i, b in enumerate(connectivity.body_ids.tolist())}
        stim_idx = [id_to_idx[int(b)] for b in stimulus_ids if int(b) in id_to_idx]
        read_idx = [id_to_idx[int(b)] for b in readout_ids if int(b) in id_to_idx]
        if not stim_idx:
            raise SystemExit(f"[{label}] none of the {len(stimulus_ids)} stimulus body-ids are in this connectivity's neuron set -- cannot run")
        if not read_idx:
            raise SystemExit(f"[{label}] none of the {len(readout_ids)} readout body-ids are in this connectivity's neuron set -- cannot run")
        print(f"  [{label}] {len(stim_idx)}/{len(stimulus_ids)} stimulus, {len(read_idx)}/{len(readout_ids)} readout neuron(s) present in this connectivity")

        params = LIFParams()  # same placeholder defaults run_full_connectome_check.py uses -- Step 3's own result is what calibrates these for real, not this script
        n_activated = 0
        for trial in range(args.n_trials):
            sim = SpikingSimulator(connectivity, params, n_flies=1)
            external = torch.zeros((1, connectivity.n_neurons), dtype=torch.int64, device=device)
            external[0, stim_idx] = args.stimulus_strength
            any_readout_spike = False
            for _ in range(args.n_steps):
                spikes = sim.step(external)
                if bool(spikes[0, read_idx].any().item()):
                    any_readout_spike = True
            if any_readout_spike:
                n_activated += 1
        rate = n_activated / args.n_trials
        print(f"  [{label}] readout activated in {n_activated}/{args.n_trials} trials (rate {rate:.2f})")
        return rate

    print(f"\nRunning {args.n_trials} trial(s) x {args.n_steps} step(s) on REAL connectivity ...")
    real_rate = _run_condition(real_connectivity, "real")
    print(f"\nRunning {args.n_trials} trial(s) x {args.n_steps} step(s) on SHUFFLED connectivity ...")
    shuffled_rate = _run_condition(shuffled_connectivity, "shuffled")

    # PASS criterion: the real connectome's readout activation rate is meaningfully higher than the
    # shuffled control's -- real circuitry should propagate a stimulus to a specific downstream
    # target more reliably than a random-but-degree-matched rewiring of the same graph. This is a
    # DIRECTIONAL, not just "different", check on purpose: a shuffled graph that activates the
    # readout MORE than the real one would mean the real circuit is actively routing signal AWAY
    # from that target, which is not "no meaningful difference" -- it's evidence something else is
    # wrong (wrong readout type, wrong sign convention, wrong stimulus strength), not a pass with an
    # unusual shape.
    margin = real_rate - shuffled_rate
    passed = margin > 0.1  # 10-percentage-point margin -- arbitrary but explicit; not yet calibrated against a second run, see below

    print("\n" + "=" * 60)
    print("SUMMARY (paste this whole block back)")
    print("=" * 60)
    print(f"device: {device}")
    print(f"neurons: {real_connectivity.n_neurons:,}")
    print(f"edges: {len(weights_df):,}")
    print(f"stimulus type: {args.stimulus_type!r} (regex={args.stimulus_regex}) -- {len(stimulus_ids)} neuron(s)")
    print(f"readout type: {args.readout_type!r} (regex={args.readout_regex}) -- {len(readout_ids)} neuron(s)")
    print(f"trials: {args.n_trials}, steps/trial: {args.n_steps}, stimulus strength: {args.stimulus_strength}")
    print(f"real activation rate: {real_rate:.2f}")
    print(f"shuffled activation rate: {shuffled_rate:.2f}")
    print(f"margin (real - shuffled): {margin:+.2f}")
    print(f"PASS (margin > 0.10): {passed}")
    print(
        "\nNOTE: the 0.10 margin threshold above is a first, explicit, arbitrary guess, NOT a "
        "calibrated value -- this is this script's own first real run, so there is no prior result "
        "to calibrate it against yet. Treat the printed rates/margin as the actual finding; whether "
        "0.10 was the right bar to set is a judgment call for whoever reads this SUMMARY the first "
        "time it runs for real, and should get written down in docs/JOURNAL.md either way."
    )


if __name__ == "__main__":
    main()
