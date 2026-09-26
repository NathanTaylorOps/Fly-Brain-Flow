# data/

Nothing real lives in this directory, and nothing ever should -- see the repo root `.gitignore`'s
own "Data and recordings" section: everything under here except this file and `VERSION` is ignored.
The actual MaleCNS connectivity table, hashes, and any recordings are cloud-only, per
`docs/PLAN.md`'s own data rule -- a Kaggle dataset (or another free object store), never the laptop,
never this git repo.

`VERSION` is the one thing about the data that *does* belong in git: which exact dataset tag
everything else is built against, and the sha256 of what got pulled, so a silent upstream revision
is detectable later (see `docs/PLAN.md`'s risk table: "dataset gets revised under me").

To actually pull and verify the data yourself: `scripts/pin_dataset.py`, run on Kaggle with your
own `NEUPRINT_TOKEN` -- see that script's own docstring for exactly how and why it has to run there
rather than locally or in this sandbox.

By default it pulls only 3 of the ~11 files in the bucket -- `body-annotations`,
`body-neurotransmitters`, and the `-significant-only` `connectome-weights` variant -- the ones
Step 2's weighted-graph spiking scaffold actually needs. The full bucket is ~31GB, more than
Kaggle's free disk; pass `--files` to pull more once a later step needs synapse-point-level detail.
See `data/VERSION` for exactly which files were pulled and their hashes, and
`flybrainflow/data_config.py`'s `KNOWN_GAPS` for the one confirmed annotation gap in male-cns:v1.0.
