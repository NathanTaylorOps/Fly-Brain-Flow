#!/usr/bin/env python3
"""M1 Step 0 -- err, Step 1: pin the data. Run this once, on Kaggle (not locally, not in a
Codespace, not anywhere it might get committed) -- see docs/M1_PLAN.md's own Step 1 for why: the
pulled connectivity table and its hash file belong in a Kaggle dataset / object storage, never on
the laptop, never in this git repo (same "cloud only" rule the plan already sets for the model
weights and everything else data-sized).

What this does, in order:
  1. Connects to neuPrint with your own token (`NEUPRINT_TOKEN` -- same one from docs/SETUP.md's
     C1/C2, as a Kaggle secret this time, not a Codespaces one) and confirms `male-cns:v1.0` is
     still a dataset neuPrint actually serves -- printed, not assumed.
  2. Confirms the `flywireType` annotation column is really on MaleCNS's own neuron table (not
     just a third-party package's word for it -- see this repo's own docs/JOURNAL.md for what was
     already confirmed by desk research before this script existed, and why that's not the same
     as confirming it against the live dataset).
  3. Checks every neuron type in `flybrainflow.data_config.REQUIRED_NEURON_TYPES` is present and
     annotated, plus a broader search for Johnston's organ (JO-*) and giant-fibre-pathway neurons,
     which don't reduce to one exact type string (see that module's own comment).
  4. Downloads the bulk, pre-exported connectivity table directly from Google Cloud Storage
     (`flybrainflow.data_config.MALECNS_BULK_BUCKET`) rather than paging it out through neuPrint's
     query API one neuron at a time.
  5. Hashes every downloaded file (sha256) and writes a hash manifest alongside them.

Usage (in a Kaggle notebook cell, `NEUPRINT_TOKEN` already set as a Kaggle secret):

    !pip install neuprint-python gcsfs pyarrow
    !python pin_dataset.py --out /kaggle/working/malecns_v1_0

Prints a plain-text summary at the end -- paste that back so the actual pass/fail (not just "it
ran") gets written down in docs/JOURNAL.md and, if anything's missing, `data_config.py` gets fixed
before Step 2 builds on top of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrainflow.data_config import (
    GIANT_FIBRE_TYPES,
    JOHNSTONS_ORGAN_PREFIX,
    KNOWN_GAPS,
    LC10_SUBTYPE_PREFIX,
    MALECNS_BULK_BUCKET,
    MALECNS_VERSION,
    REQUIRED_NEURON_TYPES,
)

# The only files M1 Step 2's weighted-graph spiking scaffold actually needs -- neurons + weighted
# synapse edges. The full bucket is ~31GB (11 files, see docs/JOURNAL.md's 2026-09-26 entry for the
# real per-file sizes) which exceeds Kaggle's ~20.9GB free disk on /kaggle/working; everything else
# (body-stats, the two other connectome-weights variants, all syn-partners variants, syn-points,
# tbar-neurotransmitters) is synapse-point/spatial-level detail not needed until much later (e.g.
# M3's inspector). Override with --files if a later step needs more.
DEFAULT_FILES = (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
)


def _get_neuprint_token() -> str | None:
    """NEUPRINT_TOKEN the normal way (Codespaces/local env var) first, falling back to Kaggle's
    own Secrets API -- Kaggle does NOT inject secrets into os.environ the way Codespaces does, they
    have to be fetched explicitly via kaggle_secrets.UserSecretsClient. Confirmed live 2026-09-26:
    the exact secret name is whatever you named it in Add-ons -> Secrets (case-sensitive) -- pass
    --secret-name if you didn't call it NEUPRINT_TOKEN there.
    """
    token = os.environ.get("NEUPRINT_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None
    try:
        return UserSecretsClient().get_secret(_SECRET_NAME)
    except Exception:
        return None


_SECRET_NAME = "NEUPRINT_TOKEN"


def main() -> None:
    global _SECRET_NAME
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory to download the bulk connectivity table into")
    parser.add_argument("--skip-download", action="store_true", help="only run the neuPrint checks, skip the (large) bulk download")
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES), help="which bucket filenames to download (default: the 3 needed for Step 2, not the full ~31GB bucket)")
    parser.add_argument("--secret-name", default="NEUPRINT_TOKEN", help="Kaggle secret name to look up if NEUPRINT_TOKEN isn't in the environment (case-sensitive, default NEUPRINT_TOKEN)")
    args = parser.parse_args()
    _SECRET_NAME = args.secret_name

    token = _get_neuprint_token()
    if not token:
        parser.error(
            "NEUPRINT_TOKEN is not set and no Kaggle secret was found -- either set the "
            "NEUPRINT_TOKEN environment variable, or add it as a Kaggle secret (Add-ons -> "
            "Secrets), attach it to this notebook, and pass --secret-name if you didn't name it "
            "NEUPRINT_TOKEN there"
        )

    try:
        from neuprint import Client
    except ImportError:
        parser.error("neuprint-python isn't installed -- run: pip install neuprint-python")

    print(f"Connecting to neuPrint, expecting dataset {MALECNS_VERSION!r} ...")
    client = Client("https://neuprint.janelia.org", dataset=MALECNS_VERSION, token=token)
    available = list(client.fetch_datasets().keys())
    print(f"neuPrint reports these datasets: {available}")
    tag_ok = MALECNS_VERSION in available
    print(f"[{'OK' if tag_ok else 'FAIL'}] {MALECNS_VERSION!r} is {'' if tag_ok else 'NOT '}being served")
    if not tag_ok:
        print("STOP: the pinned tag in flybrainflow/data_config.py is stale -- update it to whatever")
        print("male-cns:* tag neuPrint actually lists above before doing anything else.")

    # -- flywireType column ------------------------------------------------
    from neuprint import NeuronCriteria, fetch_neurons

    sample_df, _ = fetch_neurons(NeuronCriteria(type="Or42b", client=client))
    has_flywire_type = "flywireType" in sample_df.columns
    print(f"[{'OK' if has_flywire_type else 'FAIL'}] 'flywireType' column {'present' if has_flywire_type else 'MISSING'} on the neuron annotation table")
    if has_flywire_type:
        non_null = sample_df["flywireType"].notna().sum()
        print(f"    ({non_null}/{len(sample_df)} sampled Or42b neurons have a non-null flywireType)")

    # -- required neuron types ----------------------------------------------
    print("\nChecking REQUIRED_NEURON_TYPES coverage:")
    missing = []
    for t in REQUIRED_NEURON_TYPES:
        df, _ = fetch_neurons(NeuronCriteria(type=t, client=client))
        n = len(df)
        print(f"  [{'OK' if n else 'MISSING'}] {t}: {n} neuron(s)")
        if n == 0:
            missing.append(t)

    # -- LC10 (prefix match, not exact -- see data_config.py's LC10_SUBTYPE_PREFIX comment) --
    lc10_df, _ = fetch_neurons(NeuronCriteria(type=f"{LC10_SUBTYPE_PREFIX}.*", regex=True, client=client))
    print(f"\n  LC10 (type matches '{LC10_SUBTYPE_PREFIX}.*'): {len(lc10_df)} neuron(s)")
    if len(lc10_df):
        print(f"    exact subtypes found: {sorted(lc10_df['type'].dropna().unique().tolist())}")

    # -- KNOWN_GAPS reminder (already confirmed absent, 2026-09-26 -- see data_config.py) --
    print(f"\n  KNOWN_GAPS (confirmed absent under type/flywireType/receptorType, not re-checked here): {list(KNOWN_GAPS)}")

    # -- Johnston's organ / giant-fibre pathway (not exact-type-string checks) --
    jo_df, _ = fetch_neurons(NeuronCriteria(type=f"{JOHNSTONS_ORGAN_PREFIX}.*", regex=True, client=client))
    print(f"\n  Johnston's organ (type matches '{JOHNSTONS_ORGAN_PREFIX}.*'): {len(jo_df)} neuron(s)")
    if len(jo_df):
        print(f"    exact types found: {sorted(jo_df['type'].dropna().unique().tolist())}")
    gf_pattern = "(" + "|".join(GIANT_FIBRE_TYPES) + ").*"
    gf_df, _ = fetch_neurons(NeuronCriteria(type=gf_pattern, regex=True, client=client))
    print(f"  Giant-fibre pathway (type matches '{gf_pattern}'): {len(gf_df)} neuron(s)")
    if len(gf_df):
        print(f"    exact types found: {sorted(gf_df['type'].dropna().unique().tolist())}")

    # -- bulk connectivity table ---------------------------------------------
    out_dir = Path(args.out)
    hashes: dict[str, str] = {}
    if args.skip_download:
        print("\n--skip-download set -- not pulling the bulk connectivity table.")
    else:
        print(f"\nDownloading {len(args.files)} file(s) from {MALECNS_BULK_BUCKET} -> {out_dir} ...")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import gcsfs
        except ImportError:
            parser.error("gcsfs isn't installed -- run: pip install gcsfs pyarrow")
        fs = gcsfs.GCSFileSystem(token="anon")  # public bucket, no GCS credentials needed
        bucket_path = MALECNS_BULK_BUCKET.removeprefix("gs://")
        # Only the files named in --files (default: DEFAULT_FILES, the 3 actually needed) -- NOT
        # fs.ls(bucket_path), which would pull the whole ~31GB bucket and blow Kaggle's ~20.9GB
        # free disk (confirmed live 2026-09-26 -- see docs/JOURNAL.md for the real per-file sizes).
        for name in args.files:
            remote = bucket_path + name
            local = out_dir / name
            print(f"  {remote} -> {local}")
            fs.get(remote, str(local))
            h = hashlib.sha256()
            with open(local, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            hashes[name] = h.hexdigest()
        manifest_path = out_dir / "sha256_manifest.json"
        manifest_path.write_text(json.dumps({"dataset": MALECNS_VERSION, "files": hashes}, indent=2))
        print(f"  Wrote hash manifest -> {manifest_path}")

    print("\n" + "=" * 60)
    print("SUMMARY (paste this whole block back)")
    print("=" * 60)
    print(f"dataset tag confirmed: {tag_ok} ({MALECNS_VERSION})")
    print(f"flywireType column present: {has_flywire_type}")
    print(f"missing required types (unexpected -- investigate): {missing or 'none'}")
    print(f"known gaps (expected absent, see data_config.KNOWN_GAPS): {list(KNOWN_GAPS)}")
    print(f"LC10 subtypes found: {sorted(lc10_df['type'].dropna().unique().tolist()) if len(lc10_df) else 'NONE FOUND'}")
    print(f"Johnston's organ types found: {sorted(jo_df['type'].dropna().unique().tolist()) if len(jo_df) else 'NONE FOUND'}")
    print(f"giant-fibre-pathway types found: {sorted(gf_df['type'].dropna().unique().tolist()) if len(gf_df) else 'NONE FOUND'}")
    print(f"files hashed: {list(hashes.keys())}")


if __name__ == "__main__":
    main()
