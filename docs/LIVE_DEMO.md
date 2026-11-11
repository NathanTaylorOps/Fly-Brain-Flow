# Live MaleCNS experiment

This implements a running real-connectome experiment inside Fly-Brain-Flow. It is
not completion of M1's moved-food navigation gate or M2's crowd benchmark.

## Start

Use Python 3.11 and install `pip install -e ".[dev,live]"`. For the optional Brian2
reference test also install `pip install -e ".[reference]"`.

Supply an existing `public/data` bundle from Lulzx/fly-brain, reviewed at commit
`ba0b4f815c00e81a2ef1f36165df2b1891e5e39c`. Required files: `meta.json`,
`neurons.bin`, `graph_w3.bin`, `ntsign.bin`. Data is separate from this source archive;
do not replace it with a toy graph and retain the real-data label.

```text
python scripts/prepare_live_data.py --bundle /path/to/public/data
python scripts/live_demo.py --data-dir /path/to/public/data --seed 7
```

Open http://127.0.0.1:8765/. Without `--data-dir`, the service explicitly runs ToyBrain.
The local service is a single shared session, not a production multi-user host.

## Demonstrate

1. Click Run and allow the single fly to appear. With no stimulation it remains quiet.
2. Select Visual escape and set external drive to 100 Hz. The measured LC4/LPLC2
   neurons drive the neural graph. DNp01 responds and the declared 2D decoder moves the fly.
3. Silence the displayed readouts. Their activity and the fly's movement stop.
4. Reset. Select Sugar taste and apply 100 Hz. Inspect MN9. Feeding-neuron activity
   is distinct from walking; taste stimulation need not move the fly.
5. Experimental odor feedback samples the actual arena at two antenna positions,
   then drives general left/right ORNs. It is not a validated receptor-specific
   model and has not established food navigation.
6. Click inside the walkable rectangle to move food. Save experiment downloads
   observed states and interventions, not a complete neural checkpoint.

## Evidence and limits

The graph contains 165,122 traced neurons and 10,511,038 directed neuron-pair
connections with at least three synapses. Every loaded edge is retained by the engine.
Signs and annotations are inherited from the bundle, including uncertain fractional
signs. Hashes identify the four actual input files. Neither the graph nor these
annotations are asserted to match the complete original MaleCNS export.

The engine uses 0.1 ms steps, exact linear subthreshold integration, a delayed spike
queue, explicit refractory handling, seeded Poisson stimulation and persistent state.
It shares connectivity across independent per-agent states. This is a numerical
implementation of assumptions, not a claim of complete fly physiology.

The 2D decoder is engineered: DNp09 supplies forward drive, DNp01 an illustrative
escape impulse, DNa02 left-right difference supplies yaw. It has no food vector,
path planner or direct sensory-to-movement bypass. Saturating these output rates at
the arena's maximum speed is a presentation convention, not measured biomechanics.

The browser displays a sample of measured soma coordinates; missing somas are omitted
from that view only. Bright points are the most active reported cells. The full loaded
neural graph still simulates. Neural rates are measured over the last step and can be
noisy. Giant-fiber rates are much higher than a biological escape event; this is not
physiological validation.

```text
pytest
python scripts/validate_live.py --data-dir /path/to/public/data --out validation.json
```

Validation includes three seeds, rest, taste, visual stimulation, readout silencing,
and a target-shuffled graph. The shuffle preserves each neuron's in/out edge count
and each source's signed weight multiset; it does not preserve spatial structure or
each target's total synaptic input weight. Passing a readout lesion is not sufficient
evidence of a biologically correct circuit or natural navigation.

## Next decisions

1. Resolve the forward-walking response and sensory encoding before claiming M1.
2. Establish moved-target success on multiple seeds with no-input, shuffled and
   replay controls. Record failures as well as successes.
3. Load the intended unfiltered MaleCNS graph and repeat the experiments.
4. Benchmark the chosen CPU/GPU under active workloads before increasing population.
5. Add isolation, compute limits and hosting configuration before public deployment.

The working prototype deliberately makes these gaps visible so subsequent work
targets the actual failures rather than replacing them with scripted behavior.
