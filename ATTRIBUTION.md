# Attribution

Everything this project is built on, with licences. If I've used it, it's here.

## Data

| What | Who | Licence | Used for |
|---|---|---|---|
| **MaleCNS** — complete male *Drosophila* central nervous system connectome (v1.0; exact version and file hashes to be pinned in `data/VERSION` at first data pull) | Berg S. et al. (2025), *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, bioRxiv 2025.10.09.680999. Janelia FlyEM (HHMI), University of Cambridge, MRC LMB, Google Research. https://male-cns.janelia.org/ | CC-BY 4.0 | The brain. Every agent. |
| **FlyWire** — complete female *Drosophila* brain connectome | Dorkenwald S. et al. (2024), *Neuronal wiring diagram of an adult brain*, Nature 634:124–138; Schlegel P. et al. (2024), *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature 634:139–152. https://codex.flywire.ai/ | CC-BY-NC 4.0 | Calibration only — reproducing the published sugar → feeding result before trusting my own build. Not redistributed. Not in the shipped project. |

## Code

| What | Who | Licence | Used for |
|---|---|---|---|
| Drosophila_brain_model | Shiu P. K. et al. (2024), *A Drosophila computational brain model reveals sensorimotor processing*, Nature 634:210–219. https://github.com/philshiu/Drosophila_brain_model | see repo | The spiking (LIF) formulation and the direct-injection approach to sensory input |
| flyvis | Lappalainen J. K. et al. (2024), *Connectome-constrained networks predict neural activity across the fly visual system*, Nature. https://github.com/TuragaLab/flyvis | see repo | Reference for the rate-based formulation |
| fly-brain (embodied *Drosophila*) | https://github.com/erojasoficial-byte/fly-brain | MIT | Reference for the brain-to-body bridge and GPU batching. **Code only** — it is built on FlyWire, so any data derived through it is CC-BY-NC and is not used here. |
| PySocialForce | https://github.com/yuxiang-gao/PySocialForce | MIT | Algorithmic reference for the baseline model's steering (goal-seeking plus exponential pedestrian/wall repulsion); reimplemented against this project's own map and physics rather than imported, so every agent type moves through the same shared solver — see `flybrainflow/brains/baseline.py` |
| Connectome Interpreter | YijieYin/connectome-interpreter | see repo | Pathway tracing for the "why did it turn here?" inspector layer |
| neuprint-python, navis, fafbseg, osmnx | respective authors | see repos | Data access and map loading |

## Method references

The steering, avoidance, wind and odour circuits I'm reading from and driving are taken from the published literature listed in [docs/PLAN.md](docs/PLAN.md#reading-list). Nothing in the neuron mappings is original; the choice of which published mapping to use is mine.

## This project

Code: MIT — see [LICENSE](LICENSE).

Planned and drafted with AI assistance; the decisions are mine.
