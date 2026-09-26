"""Pinned dataset versions. One place, not a magic string scattered across the codebase --
`docs/M1_PLAN.md`'s own Step 1 names this as the fix for PLAN.md's "dataset gets revised under me"
risk: if every consumer imports `MALECNS_VERSION` from here instead of typing `"male-cns:v1.0"`
directly, a version bump (or a rollback) is a one-line change, not a grep-and-replace across
however many places ended up needing it.

Confirmed against Janelia's own MaleCNS project page (male-cns.janelia.org) on 2026-09-26:
v1.0 was released 2026-06-08. The project page's "accompanying publication" date, 2026-09-03,
is NOT the same date as the Berg et al. bioRxiv preprint ATTRIBUTION.md cites (2025.10.09.680999)
-- unclear whether 2026-09-03 is that same paper's later peer-reviewed journal publication date, or
a different publication entirely. Not resolved; check before citing "the publication" anywhere
that needs to be precise about which one. male-cns:v1.0 is still the current, latest tag as of this
check either way. Re-confirm this against neuPrint directly (`Client(...).fetch_datasets()`, see
`scripts/pin_dataset.py`) before trusting it blindly for anything that matters, since a desk check
today doesn't prove it hasn't moved since.
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
# field -- correct for every name below except LC10, which needs prefix matching (see below), and
# the four ORN/GRN types documented as a confirmed gap in KNOWN_GAPS below.
REQUIRED_NEURON_TYPES = (
    "Or42b",  # sugar/food odour (ORN) -- CONFIRMED GAP, see KNOWN_GAPS
    "Gr64f",  # sugar taste (GRN) -- CONFIRMED GAP, see KNOWN_GAPS
    "Gr5a",  # sugar taste (GRN) -- CONFIRMED GAP, see KNOWN_GAPS
    "Or47b",  # mate-relevant odour (ORN) -- CONFIRMED GAP, see KNOWN_GAPS
    "LC16",  # visual, looming/avoidance relevant
    "DNa02",  # descending, steering
    "DNa01",  # descending, steering
    "PFL2",  # central complex, steering
    "PFL3",  # central complex, steering
    "DNp09",  # descending, forward speed
    "MDN",  # descending, backward locomotion (cut for v1, still checked for completeness)
    "MBON32",  # mushroom body output, learned valence
)

# LC10 is NOT an exact-match miss -- it's real, just split across 7 confirmed subtypes rather than
# one bare "LC10" string: LC10_unclear, LC10a, LC10b, LC10c-1, LC10c-2, LC10d, LC10e (960 neurons
# total, confirmed live against neuPrint 2026-09-26). Anything that needs "LC10" (visual,
# mate-seeking) should match this prefix, not REQUIRED_NEURON_TYPES' exact-match list above.
LC10_SUBTYPE_PREFIX = "LC10"  # use e.g. `type.str.startswith(LC10_SUBTYPE_PREFIX)`, not `type ==`

# NEURON-COUNT RECONCILIATION (2026-09-26, checked against the downloaded body-annotations
# table): the annotation table has 211,577 rows total, not the 166,691 PLAN.md/README.md cite from
# Janelia's published headline count -- resolved by the table's own `status` column: 165,122 rows
# are `status == "Traced"` (within ~1% of the published figure -- the small residual gap is likely
# just dataset drift between whenever Janelia's project page was last updated and this pull). The
# other 46,455 rows are `Orphan` (15,925), `Glia` (11,864), `Unimportant` (10,751), `None` (5,472),
# `Assign` (1,832), and `Anchor` (611) -- non-neuron or partially-traced bodies the raw table
# includes that Janelia's "neurons" headline count doesn't. Anything that needs "the neuron count"
# should filter on `status == TRACED_STATUS`, not use the raw row count.
TRACED_STATUS = "Traced"  # annotations["status"] == TRACED_STATUS -- the ~166.7k "real" neurons

# CONNECTIVITY TABLE, confirmed 2026-09-26 against the downloaded
# connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather: 25,568,639 weighted edges
# (close to but not identical to the 24.5M a third-party side project reports for the same file --
# plausibly a threshold/version difference, not investigated further), columns `body_pre`,
# `body_post`, `weight` (int, observed range 1-2591, mean ~4.85), `type_pre`, `type_post`. Every
# body-id in this table (163,663 unique `body_pre`, 164,607 unique `body_post`, 164,740 combined)
# was confirmed present in the annotation table -- 0% missing, the two files join cleanly on body
# ID at the same confidence threshold, so Step 2 can build directly on both without a reconciliation
# step of its own.
#
# CONFIRMED GAP (checked live against neuPrint + the downloaded body-annotations table,
# 2026-09-26): Or42b, Gr64f, Gr5a, and Or47b do not exist
# in male-cns:v1.0 under any of `type`, `flywireType`, or `receptorType` -- checked all three,
# zero matches on every one. This is not a search-string problem. The antennal nerve ("AN") DOES
# appear in the `entryNerve` column, meaning antennal-lobe input fibers are physically present in
# this CNS-only EM volume -- but `receptorType` carries only three unrelated values
# (putative_IR52b, putative_ppk23, putative_ppk25), so MaleCNS's own annotation pipeline never
# assigned these axon terminals a receptor-gene identity the way hemibrain/FlyWire do. The fibers
# are there; the fine-grained labels for them are not.
#
# Substitution / resolution is a Step 4 (sensory model) decision, not a Step 1 one -- Step 1's job
# was confirm-or-document, and this is now documented. The live options when Step 4 arrives:
#   (a) cross-reference hemibrain or FlyWire (which DO carry these exact types) by antennal-lobe
#       glomerulus identity, and inject synthetic input at the matching MaleCNS glomerulus/PN;
#   (b) drop these four from the sensory model and use MaleCNS's own coarser antennal-lobe
#       projection-neuron/local-interneuron annotations as the food-odour/taste input proxy instead;
#   (c) accept the fibers are present but untyped, and treat "food odour" / "sugar taste" input as
#       an unnamed antennal-lobe glomerulus rather than a named receptor -- weakest fidelity, least
#       extra work.
# None of these is picked yet -- flag for Nathan when Step 4 planning starts.
KNOWN_GAPS = ("Or42b", "Gr64f", "Gr5a", "Or47b")

# PLAN.md also names "Johnston's organ" (the antennal mechanosensory organ that senses wind/sound,
# via several JO-* subtype neurons, not one exact type string) and "the giant-fibre pathway" (the
# escape circuit, several specifically-named cells rather than one type) -- neither maps cleanly
# onto a single `type ==` check the way `REQUIRED_NEURON_TYPES` above does. Confirmed live against
# neuPrint 2026-09-26:
#   Johnston's organ (type matches "JO-.*"): 672 neurons, 33 exact subtypes -- JO-A-unclear, JO-A1,
#     JO-A2, JO-A3, JO-A4, JO-B-unclear, JO-B1_a, JO-B1_b, JO-B1_c, JO-B2, JO-B3, JO-B4_a, JO-B4_b,
#     JO-CA1, JO-CA2, JO-CL, JO-CM, JO-DA, JO-DP, JO-ED1, JO-ED2_a, JO-ED2_b, JO-ED2_c, JO-EV1,
#     JO-EV2, JO-EV3, JO-EV4, JO-EV5, JO-EV6, JO-FD1, JO-FD2, JO-FV, JO-mz, JO-unclear.
#   Giant-fibre pathway (type matches "(GF|PSI|TTMn).*"): 38 neurons -- GFC1, GFC2, GFC3, GFC4,
#     PSI, TTMn.
JOHNSTONS_ORGAN_PREFIX = "JO-"
GIANT_FIBRE_TYPES = ("GFC1", "GFC2", "GFC3", "GFC4", "PSI", "TTMn")
