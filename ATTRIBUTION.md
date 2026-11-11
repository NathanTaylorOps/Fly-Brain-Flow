# Attribution

Everything this project is built on, with licences. If I've used it, it's here.

## Data

| What | Who | Licence | Used for |
|---|---|---|---|
| **MaleCNS** â€” complete male *Drosophila* central nervous system connectome (v1.0; version and file hashes pinned in `data/VERSION`) | Berg S. et al. (2025), *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, bioRxiv 2025.10.09.680999. Janelia FlyEM (HHMI), University of Cambridge, MRC LMB, Google Research. https://male-cns.janelia.org/ | CC-BY 4.0 | The brain. Every agent. |
| **FlyWire** â€” complete female *Drosophila* brain connectome | Dorkenwald S. et al. (2024), *Neuronal wiring diagram of an adult brain*, Nature 634:124â€“138; Schlegel P. et al. (2024), *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature 634:139â€“152. https://codex.flywire.ai/ | CC-BY-NC 4.0 | Calibration only â€” reproducing the published sugar â†’ feeding result before trusting my own build. Not redistributed. Not in the shipped project. |

## Code

| What | Who | Licence | Used for |
|---|---|---|---|
| Drosophila_brain_model | Shiu P. K. et al. (2024), *A Drosophila computational brain model reveals sensorimotor processing*, Nature 634:210â€“219. https://github.com/philshiu/Drosophila_brain_model | see repo | The spiking (LIF) formulation and the direct-injection approach to sensory input |
| flyvis | Lappalainen J. K. et al. (2024), *Connectome-constrained networks predict neural activity across the fly visual system*, Nature. https://github.com/TuragaLab/flyvis | see repo | Reference for the rate-based formulation |
| fly-brain (embodied *Drosophila*) | https://github.com/erojasoficial-byte/fly-brain | MIT | Reference for the brain-to-body bridge and GPU batching. **Code only** â€” it is built on FlyWire, so any data derived through it is CC-BY-NC and is not used here. |
| PySocialForce | https://github.com/yuxiang-gao/PySocialForce | MIT | Algorithmic reference for the baseline model's steering (goal-seeking plus exponential pedestrian/wall repulsion); reimplemented against this project's own map and physics rather than imported, so every agent type moves through the same shared solver â€” see `flybrainflow/brains/baseline.py` |
| Connectome Interpreter | YijieYin/connectome-interpreter | see repo | Pathway tracing for the "why did it turn here?" inspector layer |
| neuprint-python, navis, fafbseg, osmnx | respective authors | see repos | Data access and map loading |

## Method references

The steering, avoidance, wind and odour circuits I'm reading from and driving are taken from the published literature listed in [docs/PLAN.md](docs/PLAN.md#reading-list). Nothing in the neuron mappings is original; the choice of which published mapping to use is mine.

## This project

Code: MIT â€” see [LICENSE](LICENSE).

Planned and drafted with AI assistance; the decisions are mine.
# Live experiment implementation (November 2026)

The live CPU engine adapts the MIT-licensed active-set implementation from
[NewYorkImperialist/fly-simulator](https://github.com/NewYorkImperialist/fly-simulator),
revision `ab8d479c524ac31710a78b76244bf5e5a5bbc44d`, copyright 2026 Jayden Lin.
Its license notice is preserved in `flybrainflow/spiking/live_engine.py`.

The optional binary reader supports the format and filtered MaleCNS export from
[Lulzx/fly-brain](https://github.com/Lulzx/fly-brain), reviewed revision
`ba0b4f815c00e81a2ef1f36165df2b1891e5e39c`. Dataset assets are not included in this
repository. Credit MaleCNS (Janelia / Cambridge / MRC LMB / Google Research,
CC-BY) and the export author separately. The bundle retains 165,122 neurons and
10,511,038 connections with at least three synapses, not the unfiltered graph.
Bundled transmitter signs and annotations are inherited assumptions. The live
experiment does not import the donor's body controller or fitted physiology.\n\n