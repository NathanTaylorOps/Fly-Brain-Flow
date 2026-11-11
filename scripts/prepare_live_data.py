"""Inspect an existing MaleCNS bundle, optionally saving a normalized cache.

No downloads. Bulk data remains in the specified directory, outside this repo.
Example: python scripts/prepare_live_data.py --bundle /path/to/Lulzx/public/data
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrainflow.live_data import load_lulzx_bundle, save_live_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', help='Optional normalized .npz in your data storage directory')
    args = parser.parse_args()
    graph = load_lulzx_bundle(args.bundle)
    if args.out:
        save_live_npz(graph, args.out)
    print(json.dumps({'metadata': graph.metadata,
                      'populations': {k: len(v) for k, v in graph.populations.items()}}, indent=2))


if __name__ == '__main__':
    main()
