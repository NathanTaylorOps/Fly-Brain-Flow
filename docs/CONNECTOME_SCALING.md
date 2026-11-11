# Scaling Fly-Brain-Flow: GPU code, game technology and honest approximations

Research date: 11 November 2026. Recommendations distinguish inspected documentation/code from performance measured on this project.

## Decision

**Use a specialized batched neural engine for many independent connectomes. Use game/crowd technology for rendering and collision avoidance. These solve different bottlenecks.**

The fastest credible path is to finish today's responsive crowd application with a clearly identified controller, retain a real-connectome experiment as the reference, then benchmark a GPU batch backend before claiming hundreds of live full brains. A reduced or distilled controller may reach the visual scale sooner, but must be labeled and compared against the reference.

## Best candidates

| Candidate | What is verified | Integration judgment |
|---|---|---|
| [GeNN 5](https://genn-team.github.io/genn/documentation/5/building_networks.html) | Official documentation supports batched networks, shared parameters/connectivity and separately controlled state sharing. Custom neuron/synapse code is supported. [Variable-access documentation](https://github.com/genn-team/genn/blob/master/docs/custom_models.rst) | Best general framework candidate: share the static graph, retain independent voltage, synaptic state, delay queues, inputs and RNG per fly. Explicitly configure immutable weights as shared; do not accidentally share changing state. Custom reference dynamics and graph storage need validation. |
| [annel0/flybrain](https://github.com/annel0/flybrain) | Inspected Triton/CSR implementation, compact spikes, index-based delay queues and batched kernels. Maintainer reports 8.9× aggregate real time at batch 4 on RTX 3060, fp16; not independently reproduced here. | Best specialized kernel donor for this graph. Aggregate 8.9× divided over four simultaneous brains is about 2.2× wall-time speed per brain; it does not establish hundreds at real time. Add spike-buffer overflow detection and validate reduced precision. MIT code. |
| [Brian2CUDA](https://github.com/brian-team/brian2cuda) | Official project generates C++/CUDA from Brian2 models. GPLv3. | Useful reference-compatible GPU alternative. The standalone build/run workflow and live stimulus exchange require more work than the current Python adapter. Do not assume it supplies GeNN-style graph-sharing batches. |
| [NEST GPU](https://github.com/nest/nest-gpu) | GPU spiking-network framework, Python/C/C++ interfaces. [Official multi-GPU documentation](https://nest-gpu.readthedocs.io/en/latest/guides/multigpu_simulations.html) supports MPI distribution. | Useful if a benchmark shows multi-GPU scale is needed. Model semantics, graph replication and cluster setup must be assessed. More infrastructure than today's first integration; GPLv2 project. |
| [BrainState/JAX fly model](https://github.com/chaobrain/drosophila_whole_brain_snn_simulation) | Existing FlyWire LIF implementation with named stimuli and JAX GPU execution in primary repository. | Alternate compiled Python route. Measure batched sparse operations and host/device exchange; no verified hundred-fly capacity here. |
| [NeuroGPU](https://github.com/roybens/NeuroGPU) | Project targets NEURON models on GPU. | Different modeling stack from our point-neuron LIF. No evidence found that it is a quicker drop-in for repeated fly connectomes. Lower priority than GeNN/Triton. |
| Current Numba CPU engine | Already integrated locally; avoids a GPU install and supports persistent state. | Keep as fallback and numerical reference. Useful for development and measured small batches, without assuming CPU throughput scales to hundreds. |

The locally inspected `sazulai/Drosophila_brain_model/model_genn.py` is a useful porting example: it implements batched trials and the important refractory-input correction. Its chosen sparse allocation caps fan-out and removes approximately 1.12% of edges. A fresh GitHub file fetch returned 404 during this check; the local inspected snapshot exists. Do not depend on continued remote availability or import its pruning as an invisible optimization. GeNN's current [license file](https://github.com/genn-team/genn/blob/master/LICENSE) is LGPL-2.1.

## What game projects actually save

- [RVO2/ORCA](https://github.com/snape/RVO2): C++ collision avoidance taking preferred agent velocities, with OpenMP parallelism. Good fit for a downstream collision layer. It changes motion decisions, so display its role and compare against the same layer in both neural and baseline runs.
- [JuPedSim](https://www.jupedsim.org/stable/pedestrian_models/): existing pedestrian models and [runtime simulation API](https://www.jupedsim.org/stable/api/jupedsim/index.html) can improve evacuation baselines, geometry and crowd quality. Its routing/avoidance must not be attributed to brain wiring.
- [Unity Entities Graphics](https://docs.unity.com/en-us/engine/6000.0/manual/packages-list/packages-all/pack-safe/com-unity-entities-graphics): instancing and level-of-detail rendering; [official ECS samples](https://github.com/Unity-Technologies/EntityComponentSystemSamples) cover the wider game stack. Useful for a later 3D product, but migrating today's browser/Python application would cost time. Drawing hundreds of agents is much cheaper than simulating hundreds of 165k-neuron networks.
- The current spatial-neighbor crowd implementation and canvas rendering should be profiled first. A game-engine migration cannot fix an unmeasured neural bottleneck.

## Why graph sharing matters, and what it does not solve

These are transparent arithmetic estimates using our loaded filtered graph: 165,122 neurons and 10,511,038 weighted neuron-pair connections. They are not GPU benchmarks.

- One graph using 32-bit destination indices plus 32-bit weights needs approximately **84 MB**, plus row pointers and other metadata. Storing 500 separate graphs would require about **42 GB** before any neural state. Sharing one immutable graph avoids almost all of that duplication.
- Four float32 state arrays per fly would require about **2.64 MB per fly**, **264 MB for 100**, or **1.32 GB for 500**. Actual engines also need spike queues, refractory state, RNG, scratch space and recording.
- A dense 18-slot float32 delay buffer adds approximately **11.9 MB per fly**. Compact delayed spike lists can substantially reduce that, depending on activity.
- At 0.1 ms resolution, a dense neuron update performs about **1.65 billion neuron-steps per simulated second per fly**. One hundred brains require roughly **165 billion**, and 500 roughly **826 billion**, before synaptic delivery.
- At a hypothetical network-wide average of 1 Hz, the current graph produces roughly **10.5 million edge deliveries per second per fly**. At 100 flies that is about **1.05 billion**. Activity level is therefore central to performance.

A graph that fits in memory does not necessarily run in real time. GeNN sparse row padding can also exceed ideal CSR estimates on heavy-tailed out-degree distributions; inspect allocation before selecting a layout.

## Approximation ladder

| Option | Likely speed benefit | What must be validated/labeled |
|---|---|---|
| Exact same model, shared graph and batched independent state | Removes duplication and exposes parallel work; no automatic linear speedup | Same equations, timestep, graph, input semantics and independent RNG; compare single versus batched traces/statistics |
| Compact events, fused updates, fewer host transfers | Avoids unnecessary work and transfer overhead | No dropped spikes, correct delays, deterministic small fixtures |
| fp16 state or larger timestep | Can reduce traffic/step count | Threshold crossings, firing rates, response ordering and silencing effects may change; disclose numerical approximation |
| Sensorimotor subnetwork with recurrent pathways retained | Potentially much smaller network | Document omitted neurons/edges; validate held-out stimuli and perturbations against full graph |
| Cell-type/population rate model derived from connectivity | Much fewer state variables and coarser updates | Aggregation loses individual-cell spikes and dynamics. Label “connectome-derived reduced model,” not “full brain” |
| Distilled recurrent controller trained on reference input/output sequences | Potentially cheapest independently stateful controller for hundreds of agents | Separate hidden state per agent; hold out worlds, odors, timing and perturbations. Label “surrogate”; it cannot support arbitrary neural intervention claims unless validated |
| Small number of real brains plus inexpensive crowd controllers | Fastest mixed demonstration | Make each agent's controller visible. Never call all agents full-connectome simulations |

A reduced model is not “close” merely because motion looks similar. Compare stimulus-response curves, motor-output distributions, recovery times, perturbation effects and closed-loop trajectories. If the reference cannot solve the navigation task, training a surrogate from it does not solve that scientific gap.

On-demand full-brain inspection also requires state history: replay from a saved checkpoint and sensory inputs, or admit an approximate state reconstruction. Starting an arbitrary fresh full brain when a user clicks a surrogate agent does not reveal that agent's historical neural state.

## Concrete next implementation sequence

1. Preserve the current model's graph fingerprint, parameters, reference stimuli and causal checks.
2. Build one persistent GPU batch service; exchange compact sensory rates and motor readouts with the arena at 20–50 Hz, while retaining the neural 0.1 ms inner timestep. Avoid full-voltage or spike-history downloads on every browser frame.
3. Test GeNN and the existing Triton donor on the same retained graph. Resolve sparse row-padding without silently pruning edges.
4. Benchmark batch sizes 1, 4, 16, 32, 64, 128, then increase only if memory and latency permit. Exercise quiet, typical and highly active inputs.
5. Record device/driver, graph counts, timestep, precision, compile/warmup cost, steady-state real-time factor, peak VRAM, dropped-spike count and p95 control latency. A live batch needs all brains to advance at least one simulated second per wall second, not just a high aggregate throughput figure.
6. If full brains miss the target, build the smallest explicitly labeled approximation that passes the chosen held-out comparisons. Keep a real reference agent visible for side-by-side inspection.
7. Scale the crowd collision/rendering stack only if separate profiling identifies it as the next bottleneck.

## Hardware verified in this session

An approved read-only Windows hardware query subsequently verified **Intel(R) Iris(R) Xe Graphics**, driver **32.0.101.7085**. No NVIDIA device was listed. The installed environment reports PyTorch **2.14.0+cpu**, `cuda.is_available() == False`, and zero CUDA devices visible to PyTorch; `nvidia-smi` is not on PATH. Do not interpret Windows `AdapterRAM` as a reliable measure of dedicated graphics memory for this integrated GPU.

**There is no usable CUDA environment in the current setup.** The proposed CUDA implementations need a compatible NVIDIA GPU locally or in the cloud, or an explicitly supported alternative backend. Intel graphics may support other compute APIs, but no equivalent connectome backend or throughput was verified here. This changes the immediate execution choice, not the need to benchmark before claiming hundreds of full brains.

No GPU package was installed and no cloud GPU was provisioned during this research. A hundred-full-brain real-time commitment remains conditional on an actual hardware benchmark.\n