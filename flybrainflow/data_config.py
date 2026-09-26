"""Pinned dataset versions. One place, not a magic string scattered across the codebase --
`docs/M1_PLAN.md`'s own Step 1 names this as the fix for PLAN.md's "dataset gets revised under me"
risk: if every consumer imports `MALECNS_VERSION` from here instead of typing `"male-cns:v1.0"`
directly, a version bump (or a rollback) is a one-line change, not a grep-and-replace across
however many places ended up needing it.

Confirmed against Janelia's own MaleCNS project page (male-cns.janelia.org) on 2026-09-26:
v1.0 was released 2026-06-08, with the accompanying publication on 2026-09-03 -- still the current,
latest tag as of this check. Re-confirm this against neuPrint directly
(`Client(...).fetch_datasets()`, see `scripts/pin_dataset.py`) before trusting it blindly for
anything that matters, since a desk check today doesn't prove it hasn't moved since.
"""

from __future__ import annotations

# The exact neuPrint dataset string every `neuprint.Client(..., dataset=...)` call should use --
# never hand-typed elsewhere.
MALECNS_VERSION = "male-cns:v1.0"

# Where the bulk, pre-exported connectivity table actually lives -- Janelia publishes flat
# (neuron + weighted-synapse-edge) tables as Arrow Feather files directly in Google Cloud Storage,
# which is what `docs/M1_PLAN.md`'s Step 1 means by "bulk-export path, don't query per neuron":
# this is faster and cheaper than paging the same data out through neuPrint's own query API.
MALECNS_BULK_BUCKET = "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/"

# The neuron *types* PLAN.md's sensory/motor tables name as needed for the real brain (Step 4) --
# checked for presence/annotation coverage in male-cns:v1.0 as part of Step 1, before anything is
# built assuming they exist. See `scripts/pin_dataset.py`. Exact-match on the `type` annotation
# field -- correct for every name below except the two noted separately.
REQUIRED_NEURON_TYPES = (
    "Or42b",  # sugar/food odour (ORN)
    "Gr64f",  # sugar taste (GRN)
    "Gr5a",  # sugar taste (GRN)
    "Or47b",  # mate-relevant odour (ORN)
    "LC10",  # visual, mate-seeking relevant
    "LC16",  # visual, looming/avoidance relevant
    "DNa02",  # descending, steering
    "DNa01",  # descending, steering
    "PFL2",  # central complex, steering
    "PFL3",  # central complex, steering
    "DNp09",  # descending, forward speed
    "MDN",  # descending, backward locomotion (cut for v1, still checked for completeness)
    "MBON32",  # mushroom body output, learned valence
)

# PLAN.md also names "Johnston's organ" (the antennal mechanosensory organ that senses wind/sound,
# via several JO-* subtype neurons, not one exact type string) and "the giant-fibre pathway" (the
# escape circuit, several specifically-named cells rather than one type) -- neither maps cleanly
# onto a single `type ==` check the way `REQUIRED_NEURON_TYPES` above does. `scripts/pin_dataset.py`
# checks these two separately, by `type` prefix/class search rather than exact match, and this
# module gets updated with whatever exact type strings that search turns up once it's actually run
# against neuPrint -- deliberately left open here rather than guessed at.
