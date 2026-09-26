# Setup — accounts and tools, before any code

Follow top to bottom. Each step says why it exists and how you know it's done. Nothing here installs anything heavy on the laptop; the laptop is a browser.

Budget for the whole thing: about an hour, most of it waiting for verification emails.

Menu paths are correct as of writing and may drift. If a menu isn't where I say, the setting still exists — search the settings page for the word in bold.

---

## Phase A — the laptop (10 minutes)

**A1. Make some room on the disk.**
Why: 6 GB free is tight for a browser with a lot of tabs, and Windows update will want space. Nothing project-related will be stored here, but the machine still needs to breathe.
- Settings → System → Storage → **Cleanup recommendations**. Take whatever it offers (temporary files, old downloads, Recycle Bin).
- Turn on **Storage Sense** while you're there.
- Done when: 15 GB+ free. If it won't get there, don't fight it; move on.

**A2. Pick one browser and stay in it.**
Why: Codespaces, Kaggle, neuPrint and GitHub all remember sign-ins per browser. Splitting across two costs you re-logins forever.
- Edge or Chrome, either is fine.
- Done when: you've decided.

**A3. Nothing else.**
No Python, no Git, no VS Code, no Docker on this machine. Everything runs in the cloud. If you ever want a nicer editor than the browser tab, VS Code desktop can connect to a Codespace — but it's ~500 MB and it isn't needed.

---

## Phase B — GitHub (20 minutes)

You already have an account. This makes sure it's in shape for a public project and creates the repo.

**B1. Two-factor authentication.**
Why: a public repo with your name on it and cloud secrets attached is worth protecting.
- GitHub → Settings → **Password and authentication** → enable 2FA (authenticator app or passkey).
- Done when: GitHub shows 2FA enabled.

**B2. Codespaces spending limit.**
Why: this is the first of the hard caps in the plan. The free tier is 60 core-hours a month (as of writing). The limit stops it billing past that.
- Settings → **Billing and plans** → **Spending limits** → Codespaces → set to **$0**.
- Done when: it says $0.

**B3. Create the repository.**
- github.com/new
- Name: `fly-brain-flow`
- Public.
- **Do not** tick "Add a README", "Add .gitignore", or "Choose a license". The first commit already contains all three.
- Create.
- Done when: you're looking at an empty repo with the "quick setup" instructions.

**B4. Make the first commit — from the browser, no Git needed.**
Why: the README, plan, attribution and licence are already written. This puts them up.
- Unzip `fly-brain-flow.zip` on the laptop. You'll get a folder with `README.md`, `LICENSE`, `ATTRIBUTION.md`, `.gitignore`, a `docs/` folder and a `data/` folder.
- In the empty repo, click **"uploading an existing file"** (it's in the quick-setup text).
- Drag the *contents* of the unzipped folder in — all files and both folders together. GitHub keeps the folder structure.
- Commit message: `Planning complete: README, plan, attribution, licence`
- Commit directly to `main`.
- Done when: the repo page shows the README rendered, and `docs/` and `data/` are visible.

**B5. Turn on Issues and Projects.**
Why: the M0 task list will live in Issues so progress is visible.
- Repo → Settings → **General** → Features → tick **Issues** and **Projects**.
- Done when: an "Issues" tab appears on the repo.

**B6. Open the Codespace.**
Why: this is the development machine from now on.
- Repo → green **Code** button → **Codespaces** tab → **Create codespace on main**.
- First build takes a minute or two. You'll land in VS Code in the browser.
- In the terminal at the bottom, type `python3 --version` and press Enter. Any 3.x is fine.
- Done when: you see a Python version.

Note: the Codespace stops itself after 30 minutes idle. That's normal and it's what keeps it free. It picks up where it left off.

---

## Phase C — the brain data (15 minutes)

**C1. neuPrint account and token.**
Why: this is where MaleCNS lives. The token is how code proves it's you.
- Go to `neuprint.janelia.org` → **Login** (it uses a Google account — use whichever one you want tied to this).
- Once in, click your avatar / account name (top right) → **Account**. You'll see an **Auth Token** — a long string. Copy it.
- Done when: you have the token in your clipboard. **Don't paste it anywhere yet.**

**C2. Store the token as a Codespaces secret — never in the repo.**
Why: a token in a public repo gets scraped within minutes.
- GitHub → Settings (your account, not the repo) → **Codespaces** → **Secrets** → New secret.
- Name: `NEUPRINT_TOKEN`
- Value: paste the token.
- Repository access: select `fly-brain-flow`.
- Done when: the secret is listed. Restart the Codespace (Codespaces menu → Stop, then reopen) so it picks the secret up.

**C3. Prove the data is reachable.** *(The only commands in this whole document. They check plumbing; they're not project code.)*
In the Codespace terminal, one line at a time:

```
pip install neuprint-python
```
```
python3 -c "from neuprint import Client; import os; c = Client('https://neuprint.janelia.org', dataset='male-cns:v1.0', token=os.environ['NEUPRINT_TOKEN']); print(list(c.fetch_datasets().keys()))"
```
- Done when: it prints a list of dataset names that includes something starting with `male-cns`. If it says `KeyError: 'NEUPRINT_TOKEN'`, the secret didn't load — restart the Codespace and try again.

**C4. FlyWire Codex — can wait until M1.**
Why: only needed for the calibration step. Sign up whenever you get there.
- `codex.flywire.ai` → sign in with Google → accept the terms.
- Done when: you can see the dataset explorer.

---

## Phase D — GPU (15 minutes now; the rest later)

**D1. Kaggle.**
Why: free GPU for the benchmark, calibration and recorded runs.
- `kaggle.com` → register (Google sign-in is fine).
- Settings → **Phone verification** → verify your number. **Without this, GPUs are locked.**
- Done when: settings show phone verified.

**D2. Prove the GPU is reachable.**
- Kaggle → **Code** → **New Notebook**.
- Right-hand panel → **Session options** → **Accelerator** → pick a GPU (T4 as of writing).
- In the first cell type `!nvidia-smi` and run it (Shift+Enter).
- Done when: you see a table with a GPU name in it. Then close the notebook — GPU time only counts while a session is running.

**D3. Azure — don't sign up yet.**
Why: the free account's starting credit runs on a 30-day clock from the day you sign up. The benchmark is weeks away. Sign up the week you need it, then check whether the trial subscription is allowed GPU quota before counting on it. If it isn't, Kaggle is the plan anyway.

**D4. RunPod / Vast.ai — don't sign up yet.**
Why: only needed if Kaggle's free hours run short. Both are prepaid — you load a small balance (say $10) and that balance *is* the spending cap; they can't bill past it. Set up on the day you need it.

---

## Phase E — the working rhythm (5 minutes)

**E1. Open the M0 issues.**
Why: turns the milestone list into visible movement.
- Repo → Issues → New issue, one per line below. Title only; no description needed yet.
  - `M0: map loader (OSM + DXF/SVG)`
  - `M0: 2D collision physics`
  - `M0: spawn / feed / leave with slots and timer`
  - `M0: odour and wind fields on the walkable grid`
  - `M0: baseline social-force agents (PySocialForce fork)`
  - `M0: recorder (paths + sensory streams)`
  - `M0: viewer (Three.js, playback)`
  - `M0: toy connectome running the real code path`
  - `M0: tests — seed, bounds, population accounting`
- Done when: nine open issues.

**E2. Journal.**
`docs/JOURNAL.md` is a private working log, gitignored, not part of the public repo. Add a line whenever something moves. Three lines max per entry: what happened, what's next, what got cut.

**E3. The two rules that keep this cheap.**
- Tokens and keys go in Codespaces secrets. Never in a file. The `.gitignore` in the first commit blocks `.env` files as a backstop, not as the plan.
- GPU sessions get closed when you walk away. Kaggle's clock runs while the tab is open.

---

## Setup is done when

- [ ] GitHub 2FA on, Codespaces spending limit $0
- [ ] `fly-brain-flow` repo public, first commit visible, Issues on
- [ ] Codespace opens and shows a Python version
- [ ] `NEUPRINT_TOKEN` secret set; the C3 check prints a dataset list with `male-cns` in it
- [ ] Kaggle phone-verified; `!nvidia-smi` shows a GPU
- [ ] Nine M0 issues open
- [ ] Journal has its first entry

That's the whole environment. Next is M0, and the first thing M0 touches is the map loader.
