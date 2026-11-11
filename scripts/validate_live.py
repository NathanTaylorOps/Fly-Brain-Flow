"""Reproduce real-data neural responses and silencing; write measured evidence."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrainflow.live_data import load_lulzx_bundle
from flybrainflow.spiking.live_engine import LiveLIFEngine

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    graph = load_lulzx_bundle(args.data_dir)
    engine = LiveLIFEngine(graph.indptr, graph.indices, graph.weights)
    engine.advance(.1)  # exclude JIT from measured runs
    records = []
    for seed in (7, 42, 123):
        for condition, group, silenced in [('rest', None, False), ('sugar', 'sugar', False),
                                            ('loom', 'loom', False), ('loom_silenced', 'loom', True)]:
            engine.reset(seed)
            engine.set_poisson(graph.populations[group] if group else [], 100 if group else 0)
            engine.set_silenced(graph.populations['DNp01'] if silenced else [])
            started = time.perf_counter()
            result = engine.advance(300)
            wall = time.perf_counter() - started
            row = {'seed': seed, 'condition': condition, 'simulated_ms': 300, 'wall_s': wall,
                   'rtf': .3 / wall, 'spikes': int(result.counts.sum()),
                   'active_neurons': int(np.count_nonzero(result.counts)),
                   'rates_hz': {k: result.rate_hz(v) for k, v in graph.populations.items()}}
            records.append(row)
            print(json.dumps({'condition': condition, 'seed': seed, 'MN9_hz': row['rates_hz']['MN9'],
                              'DNp01_hz': row['rates_hz']['DNp01'], 'rtf': row['rtf']}), flush=True)
    report = {'dataset': graph.metadata, 'params': vars(engine.params), 'records': records,
              'limits': ['Filtered graph', 'Direct neural stimulation', 'Uniform LIF assumptions',
                         'No proof of natural navigation or physiological firing rates',
                         'Silencing readout demonstrates causal removal, not wiring specificity']}
    # Null control: permute postsynaptic targets across edges. Preserves each
    # neuron's in/out edge count and presynaptic signed weight multiset, not
    # spatial structure or each target's summed synaptic weight.
    targets = np.random.default_rng(99).permutation(graph.indices)
    shuffled = LiveLIFEngine(graph.indptr, targets, graph.weights)
    shuffled.advance(.1)
    report['target_shuffle_control'] = []
    for seed in (7, 42, 123):
        for group in ('sugar', 'loom'):
            shuffled.reset(seed)
            shuffled.set_poisson(graph.populations[group], 100)
            started = time.perf_counter()
            result = shuffled.advance(300)
            row = {'seed': seed, 'condition': group, 'wall_s': time.perf_counter() - started,
                   'MN9_hz': result.rate_hz(graph.populations['MN9']),
                   'DNp01_hz': result.rate_hz(graph.populations['DNp01']),
                   'spikes': int(result.counts.sum()), 'active_neurons': int(np.count_nonzero(result.counts))}
            report['target_shuffle_control'].append(row)
            print(json.dumps({'shuffled': row}), flush=True)
    report['target_shuffle_definition'] = 'Permute all postsynaptic indices; preserve in/out edge degrees and source signed weight multisets; seed=99'
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
