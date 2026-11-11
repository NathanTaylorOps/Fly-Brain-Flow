"""Read an existing, explicitly filtered MaleCNS bundle without downloading data.

The documented flat binary format comes from MIT project Lulzx/fly-brain
(scripts/algo_structures.py and scripts/connectome_ir.py). The underlying MaleCNS
data retains its CC-BY attribution. No binary assets are bundled with this module.
Weights are signed synapse counts, NOT PSP amplitudes: the engine applies its
own declared mV-per-synapse scale. This is not the full unfiltered connectome.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LiveConnectome:
    body_ids: np.ndarray
    indptr: np.ndarray  # presynaptic rows
    indices: np.ndarray  # postsynaptic columns
    weights: np.ndarray  # signed counts; no per-edge scaling fitted here
    positions: np.ndarray
    types: np.ndarray
    instances: np.ndarray
    populations: dict[str, np.ndarray]
    metadata: dict

    @property
    def n_neurons(self) -> int:
        return len(self.body_ids)

    @property
    def n_edges(self) -> int:
        return len(self.indices)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _populations(types: np.ndarray, instances: np.ndarray) -> dict[str, np.ndarray]:
    """Exact annotation selections. Empty groups remain empty, never substituted."""
    groups = {
        'sugar': np.isin(types, ['LB3b', 'LB3c']),
        'MN9': types == 'MN9',
        'loom': np.isin(types, ['LC4', 'LPLC2']),
        'DNp01': types == 'DNp01',
        'DNa02': types == 'DNa02',
        'DNp09': types == 'DNp09',
    }
    for side in ('L', 'R'):
        sided = np.char.endswith(instances, '_' + side)
        for name in ('DNa02', 'DNp09', 'MN9', 'DNp01'):
            groups[name + '_' + side] = (types == name) & sided
        # General antennal drive only: this is not a receptor-specific food odor.
        groups['ORN_' + side] = np.char.startswith(types, 'ORN_') & sided
    return {name: np.flatnonzero(mask).astype(np.int32) for name, mask in groups.items()}


def load_lulzx_bundle(directory: str | Path, *, hash_files: bool = True) -> LiveConnectome:
    """Load ``public/data`` from the reviewed Lulzx bundle in place.

    Validates file lengths, CSR orientation/bounds, identities and signed weights.
    Hashing the four source files records the actual input, not a guessed release.
    It does not run the donor's fitted physiology, steering or body controller.
    """
    directory = Path(directory)
    files = {name: directory / name for name in
             ('meta.json', 'neurons.bin', 'graph_w3.bin', 'ntsign.bin')}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing connectome asset: {path}')
    meta = json.loads(files['meta.json'].read_text(encoding='utf-8'))
    n, edges = int(meta['N']), int(meta['edges'])
    if n <= 0 or edges < 0:
        raise ValueError('Invalid graph dimensions')
    if files['graph_w3.bin'].stat().st_size != 8 + 4 * (n + 1) + 6 * edges:
        raise ValueError('Graph file length does not match metadata')
    with files['graph_w3.bin'].open('rb') as f:
        header = np.fromfile(f, dtype='<u4', count=2)
        if tuple(header) != (n, edges):
            raise ValueError('Graph header does not match metadata')
        indptr = np.fromfile(f, dtype='<u4', count=n + 1).astype(np.int64)
        indices = np.fromfile(f, dtype='<u4', count=edges).astype(np.int32)
        counts = np.fromfile(f, dtype='<u2', count=edges)
    if indptr[0] != 0 or indptr[-1] != edges or np.any(np.diff(indptr) < 0):
        raise ValueError('Invalid presynaptic CSR offsets')
    if edges and (indices.min() < 0 or indices.max() >= n):
        raise ValueError('Postsynaptic index outside neuron table')
    min_weight = int(meta['minWeight'])
    if min_weight < 1 or (edges and counts.min() < min_weight):
        raise ValueError('Edge weights conflict with declared minimum')
    if files['neurons.bin'].stat().st_size != 8 + 33 * n:
        raise ValueError('Neuron file length does not match metadata')
    with files['neurons.bin'].open('rb') as f:
        header = np.fromfile(f, dtype='<u4', count=2)
        if int(header[0]) != n:
            raise ValueError('Neuron header does not match metadata')
        body_ids = np.fromfile(f, dtype='<u8', count=n)
        positions = np.fromfile(f, dtype='<f4', count=3 * n).reshape(n, 3)
    if len(np.unique(body_ids)) != n:
        raise ValueError('Neuron body IDs are not unique')
    # Missing somas are encoded as NaN by the donor. Preserve them rather than
    # invent coordinates; a viewer must omit these points, not the neurons.
    if np.isinf(positions).any():
        raise ValueError('Infinite neuron positions')
    if files['ntsign.bin'].stat().st_size != 4 * n:
        raise ValueError('Transmitter-sign file length does not match neurons')
    signs = np.fromfile(files['ntsign.bin'], dtype='<f4')
    if not np.isfinite(signs).all() or np.any(np.abs(signs) > 1.00001):
        raise ValueError('Transmitter signs must be finite and within [-1, 1]')
    types = np.asarray(meta['types'], dtype=str)
    instances = np.asarray(meta['instances'], dtype=str)
    if types.shape != (n,) or instances.shape != (n,):
        raise ValueError('Type/instance annotations do not match neuron count')
    weights = counts.astype(np.float32)
    weights *= np.repeat(signs, np.diff(indptr))
    metadata = {
        'dataset': 'MaleCNS v1.0 — filtered export',
        'is_real_connectome': True,
        'is_unfiltered_connectome': False,
        'source_url': 'https://github.com/Lulzx/fly-brain',
        'reviewed_source_revision': 'ba0b4f815c00e81a2ef1f36165df2b1891e5e39c',
        'data_credit': 'MaleCNS: Janelia / Cambridge / MRC LMB / Google Research (CC-BY)',
        'bundle_credit': 'Lulzx/fly-brain; filtered graph and bundled annotations',
        'neuron_count': n, 'edge_count': edges,
        'synaptic_contact_count': int(counts.sum(dtype=np.uint64)),
        'missing_soma_count': int((~np.isfinite(positions).all(axis=1)).sum()),
        'minimum_synapse_count': min_weight,
        'filter_description': f'Traced neurons; neuron-pair connections with at least {min_weight} synapses',
        'sign_policy': 'Bundled per-neuron ntsign.bin, including graded uncertain signs',
        'orientation': 'CSR rows=presynaptic; indices=postsynaptic',
        'weight_units': 'signed synapse counts; PSP scale applied by engine',
        'input_caveat': 'Direct neural stimulation; general ORN drive is not natural odor transduction',
        'sha256': {name: _sha256(path) for name, path in files.items()} if hash_files else {},
    }
    return LiveConnectome(body_ids, indptr, indices, weights, positions,
                          types, instances, _populations(types, instances), metadata)


def save_live_npz(graph: LiveConnectome, path: str | Path) -> None:
    """Optional normalized cache. Write only to a user-selected data directory."""
    populations = {name: arr.tolist() for name, arr in graph.populations.items()}
    np.savez(path, body_ids=graph.body_ids, indptr=graph.indptr,
             indices=graph.indices, weights=graph.weights, positions=graph.positions,
             types=graph.types, instances=graph.instances,
             populations_json=json.dumps(populations), metadata_json=json.dumps(graph.metadata))


def load_live_npz(path: str | Path) -> LiveConnectome:
    """Load a normalized cache written by save_live_npz (no pickle execution)."""
    with np.load(path, allow_pickle=False) as a:
        populations = {k: np.asarray(v, dtype=np.int32) for k, v in
                       json.loads(str(a['populations_json'])).items()}
        return LiveConnectome(*(a[k] for k in
            ('body_ids', 'indptr', 'indices', 'weights', 'positions', 'types', 'instances')),
            populations, json.loads(str(a['metadata_json'])))
