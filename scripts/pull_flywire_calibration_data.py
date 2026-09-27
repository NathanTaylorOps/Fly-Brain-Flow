#!/usr/bin/env python3
"""Pull FlyWire connectivity + cell-type data for M1 Step 3's OTHER leg ("Reproduce Shiu et al. on
FlyWire", docs/M1_PLAN.md's Step 3, item 1) -- the leg this project could not write with any
confidence until tonight, because how to actually get FlyWire data programmatically had never been
confirmed anywhere in this codebase (docs/SETUP.md's C4 only ever documented the `codex.flywire.ai`
WEB sign-in, not a scripted pull).

**UNVERIFIED AGAINST A REAL ACCOUNT.** Written from FlyWire/Codex's and CAVEclient's own published
documentation (2026-09-27 night, while Nathan was away -- see docs/JOURNAL.md's matching entry), NOT
run against a real token, a real account, or a real network call, because none of those exist for
this project yet and this sandbox has no route to codex.flywire.ai or a CAVE server anyway. This is a
categorically bigger leap of faith than pin_dataset.py/run_full_connectome_check.py/
run_calibration_gate.py: those are unverified against real DATA but built on an already-proven
access pattern (neuPrint + a public GCS bucket) this project has used successfully twice. This
script is the FIRST use of an entirely new access path, from documentation alone. Read it critically
before trusting it, and expect the first real run to surface at least one wrong assumption --
Codex's exact `data_product` names in particular (see PRIMARY PATH below) are a guess at
plausible-looking values, not confirmed ones.

TWO ACCESS PATHS, in the order Codex's own FAQ recommends them:

  PRIMARY -- Codex's static bulk-download API. Confirmed real and documented (codex.flywire.ai/faq,
  fetched 2026-09-27): `https://codex.flywire.ai/api/download?dataset=<name>` lists available
  data products for a chosen dataset; `https://codex.flywire.ai/api/download_resource?
  data_product=<name>&dataset=<name>&api_token=<token>` downloads one as a gzipped CSV. The FAQ's
  own words: "For programmatic analysis, use Codex's static downloadable files rather than issuing
  many live app queries." `api_token` comes from https://codex.flywire.ai/account once signed in
  (the same Google sign-in docs/SETUP.md's C4 already has Nathan doing). This is structurally the
  same shape as `pin_dataset.py`'s GCS-bucket pull for MaleCNS -- a good sign this is the right path,
  not an unnecessary detour into CAVE. Codex lists **MCNS v1.0** (i.e. male-cns:v1.0, this project's
  own already-pinned dataset) as one of its five browsable datasets alongside FAFB v783 -- interesting
  but not something this script relies on; MaleCNS keeps coming from Janelia's own GCS bucket
  (`data_config.MALECNS_BULK_BUCKET`), unchanged.

  FALLBACK -- CAVEclient, for anything Codex's static exports don't cover (e.g. live proofreading
  status, if that ever turns out to matter for this gate). Confirmed real and documented
  (caveclient.readthedocs.io, fetched 2026-09-27): `pip install caveclient`, then
  `CAVEclient(datastack_name)` after a one-time `client.auth.get_new_token()` /
  `client.auth.save_token(token=...)` browser-based token flow (token saved to
  `~/.cloudvolume/secrets/cave-secret.json` by default, so it's picked up automatically after that).
  The real `datastack_name` for FlyWire's production dataset has NOT been confirmed (a plausible
  guess like "flywire_fafb_production" is exactly the kind of unverified string this project's own
  discipline says not to hardcode without a live check -- see `--datastack-name` below, no default).

Usage (wherever this actually gets run -- Kaggle, Codespaces, or Nathan's own machine; unlike
MaleCNS's bulk pull this isn't obviously GPU/Kaggle-only, since it's a small, targeted download, not
a ~500MB connectome-wide table):

    !pip install requests
    export CODEX_API_TOKEN=...   # from https://codex.flywire.ai/account

    # Step A -- discovery only (no --data-products given): lists what Codex actually offers for
    # --dataset, so the real product name(s) get picked from real output, not guessed.
    python pull_flywire_calibration_data.py --out ./flywire_pull --dataset fafb

    # Step B -- once Step A's output names the real products for connectivity + cell types:
    python pull_flywire_calibration_data.py --out ./flywire_pull --dataset fafb \\
        --data-products "<real product name>" "<real product name>"

Prints a plain-text SUMMARY block at the end, same pattern as this project's other pull scripts --
paste it back so what actually happened (not just "the script exists") gets written down.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_DOWNLOAD_LIST_URL = "https://codex.flywire.ai/api/download"
_DOWNLOAD_RESOURCE_URL = "https://codex.flywire.ai/api/download_resource"
_TOKEN_ENV_VAR = "CODEX_API_TOKEN"

# Plausible-looking Codex data_product names for connectivity and cell-type annotations --
# UNCONFIRMED, exactly like run_calibration_gate.py's own sourced-but-unconfirmed cell-type
# candidates. Printed as a hint in discovery mode, never assumed or downloaded automatically.
_PLAUSIBLE_PRODUCT_NAMES = (
    "connections",
    "connectivity",
    "cell_stats",
    "classification",
    "cell_types",
    "synapse_table",
)


def _get_codex_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get(_TOKEN_ENV_VAR)


def _list_data_products(dataset: str, token: str):
    """GET the discovery endpoint and return whatever it reports as available data products for
    `dataset`. Response shape is NOT confirmed live (no account/network access from where this was
    written) -- handles a few plausible shapes (a bare list, or a dict with a 'data_products'/
    'products' key) and fails loudly, printing the raw response, if none of those match, rather than
    silently returning an empty list that would look like "nothing available" instead of "this
    script's assumption about the response shape was wrong."
    """
    import requests

    resp = requests.get(_DOWNLOAD_LIST_URL, params={"dataset": dataset, "api_token": token}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data_products", "products", "resources"):
            if key in data:
                return data[key]
    raise SystemExit(
        f"unrecognized response shape from {_DOWNLOAD_LIST_URL} -- this script's guess at the JSON "
        f"shape (a list, or a dict with 'data_products'/'products'/'resources') doesn't match what "
        f"came back. Raw response (fix this script's _list_data_products once you see the real "
        f"shape):\n{json.dumps(data, indent=2)[:2000]}"
    )


def _download_resource(dataset: str, data_product: str, token: str, out_path: Path) -> int:
    """Downloads one gzipped-CSV data product, decompresses it, writes it to `out_path`. Returns the
    decompressed byte count. Per Codex's FAQ, this is a plain HTTPS GET with query params -- no
    special headers or multi-step flow documented."""
    import requests

    resp = requests.get(
        _DOWNLOAD_RESOURCE_URL,
        params={"dataset": dataset, "data_product": data_product, "api_token": token},
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()
    raw = resp.content
    try:
        decompressed = gzip.decompress(raw)
    except OSError:
        # Not actually gzipped -- write the raw bytes instead of failing outright, and say so, in
        # case a particular data_product turns out to be plain CSV/JSON rather than gzipped.
        print(f"    NOTE: response for {data_product!r} didn't decompress as gzip -- writing raw bytes as-is")
        decompressed = raw
    out_path.write_bytes(decompressed)
    return len(decompressed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory to write downloaded files into")
    parser.add_argument("--dataset", default="fafb", help="Codex dataset name (default: 'fafb' -- FlyWire's own connectome; 'fafb' is Codex's own short name for it per its FAQ/UI, not independently reverse-engineered here)")
    parser.add_argument("--api-token", default=None, help=f"Codex API token (from https://codex.flywire.ai/account) -- falls back to the {_TOKEN_ENV_VAR} environment variable")
    parser.add_argument("--data-products", nargs="+", default=None, help="which Codex data_product name(s) to actually download -- omit to run in discovery mode only")
    args = parser.parse_args()

    token = _get_codex_token(args.api_token)
    if not token:
        parser.error(
            f"no Codex API token -- pass --api-token or set {_TOKEN_ENV_VAR}. Get one from "
            "https://codex.flywire.ai/account after signing in (same Google sign-in docs/SETUP.md's "
            "C4 already has set up)"
        )

    try:
        import requests  # noqa: F401
    except ImportError:
        parser.error("requests isn't installed -- run: pip install requests")

    print(f"Listing available data products for dataset={args.dataset!r} ...")
    products = _list_data_products(args.dataset, token)
    print(f"  Codex reports {len(products)} data product(s):")
    for p in products:
        print(f"    {p}")

    plausible_present = [p for p in _PLAUSIBLE_PRODUCT_NAMES if p in [str(x) for x in products]]
    if plausible_present:
        print(f"\n  Of this script's own UNCONFIRMED guesses at likely names, these actually appear "
              f"in the real list above: {plausible_present} -- worth trying first, but still read "
              "the full list above rather than trusting this guess blindly.")

    if not args.data_products:
        print(
            "\nDiscovery mode only (no --data-products given) -- nothing downloaded. Re-run with "
            "--data-products \"<real name(s) from the list above>\" once you've picked the real "
            "connectivity/cell-type products this gate actually needs."
        )
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, int] = {}
    for product in args.data_products:
        out_path = out_dir / f"{product}.csv"
        print(f"\nDownloading data_product={product!r} -> {out_path} ...")
        n_bytes = _download_resource(args.dataset, product, token, out_path)
        downloaded[product] = n_bytes
        print(f"  {n_bytes:,} bytes written")

    print("\n" + "=" * 60)
    print("SUMMARY (paste this whole block back)")
    print("=" * 60)
    print(f"dataset: {args.dataset}")
    print(f"data products available: {len(products)}")
    print(f"data products downloaded: {downloaded}")
    print(f"output directory: {out_dir}")
    print(
        "\nNOTE: this script's HTTP calls, response-shape handling, and gzip assumption are all "
        "UNVERIFIED against the real Codex API as of this run being the first time it's ever "
        "executed against a real account -- if anything above looks wrong (an unexpected response "
        "shape, an auth error, a product that isn't actually gzipped), that's genuinely new "
        "information, not a sign this run failed for a trivial reason. Write down what actually "
        "happened in docs/JOURNAL.md, wrong turns included, same as every other first real run in "
        "this project."
    )


if __name__ == "__main__":
    main()
