import json
import numpy as np
import pytest
from flybrainflow.live_data import load_lulzx_bundle, load_live_npz, save_live_npz


def bundle(tmp_path):
    n, e = 3, 3
    (tmp_path / 'meta.json').write_text(json.dumps({
        'N': n, 'edges': e, 'minWeight': 3,
        'types': ['LB3b', 'MN9', 'DNa02'],
        'instances': ['LB3b_L', 'MN9_R', 'DNa02_L']}))
    with (tmp_path / 'graph_w3.bin').open('wb') as f:
        np.array([n, e], '<u4').tofile(f)
        np.array([0, 2, 2, 3], '<u4').tofile(f)
        np.array([1, 2, 0], '<u4').tofile(f)
        np.array([3, 9, 5], '<u2').tofile(f)
    with (tmp_path / 'neurons.bin').open('wb') as f:
        np.array([n, 0], '<u4').tofile(f)
        np.array([720575940000000001, 720575940000000002, 720575940000000003], '<u8').tofile(f)
        np.arange(n * 3, dtype='<f4').tofile(f)
        f.write(bytes(13 * n))
    np.array([1, 0.5, -1], '<f4').tofile(tmp_path / 'ntsign.bin')
    return tmp_path


def test_preserves_ids_direction_counts_and_signs(tmp_path):
    graph = load_lulzx_bundle(bundle(tmp_path))
    assert graph.body_ids[0] == 720575940000000001
    assert graph.indptr.tolist() == [0, 2, 2, 3]
    assert graph.weights.tolist() == [3, 9, -5]
    assert graph.metadata['synaptic_contact_count'] == 17
    assert graph.metadata['is_unfiltered_connectome'] is False
    assert len(graph.metadata['sha256']) == 4
    assert graph.populations['sugar'].tolist() == [0]
    assert graph.populations['DNa02_L'].tolist() == [2]
    assert graph.populations['DNa02_R'].size == 0


def test_cache_roundtrip_without_pickle(tmp_path):
    graph = load_lulzx_bundle(bundle(tmp_path))
    save_live_npz(graph, tmp_path / 'cache.npz')
    restored = load_live_npz(tmp_path / 'cache.npz')
    np.testing.assert_array_equal(restored.weights, graph.weights)
    assert restored.metadata == graph.metadata


def test_rejects_truncated_and_invalid_edges(tmp_path):
    folder = bundle(tmp_path)
    path = folder / 'graph_w3.bin'
    with path.open('r+b') as f:
        f.seek(8 + 4 * 4)
        np.array([10], '<u4').tofile(f)
    with pytest.raises(ValueError, match='outside'):
        load_lulzx_bundle(folder)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='length'):
        load_lulzx_bundle(folder)


def test_rejects_missing_asset(tmp_path):
    with pytest.raises(FileNotFoundError, match='meta.json'):
        load_lulzx_bundle(tmp_path)
