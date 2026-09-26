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
