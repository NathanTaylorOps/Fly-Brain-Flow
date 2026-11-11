"""Serve a live odor maze; behavior is explicit, connectome mode optional."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
import numpy as np

from flybrainflow.live_demo import LiveSession, create_server, maze_session
from flybrainflow.scenario import Scenario
from flybrainflow.sim import Sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--data-dir", type=Path, help="Existing Lulzx MaleCNS bundle; enables optional real-connectome mode, loaded on first selection")
    parser.add_argument("--controller", choices=("behavior", "connectome"), default="behavior")
    parser.add_argument("--warmup-s", type=float, default=60, help="Odor prewarm simulation seconds (not wall time)")
    args = parser.parse_args()
    factory = None
    if args.scenario:
        original = Scenario.from_toml(args.scenario)
        def factory(seed):
            sc = replace(original, meta=replace(original.meta, seed=seed))
            return Sim.from_scenario(sc, rng=np.random.default_rng(seed))
    if args.scenario:
        if args.data_dir or args.controller != "behavior":
            parser.error("Custom scenarios currently support only toy mode")
        session = LiveSession(factory, seed=args.seed)
    else:
        import math
        if not math.isfinite(args.warmup_s) or not 0 <= args.warmup_s <= 300:
            parser.error("--warmup-s must be 0..300")
        graph_cache = {}
        connectome_factory = None
        if args.data_dir:
            def connectome_factory(seed, arena):
                from flybrainflow.live_data import load_lulzx_bundle
                from flybrainflow.brains.live_connectome import LiveConnectomeBrain
                if "graph" not in graph_cache:
                    print("Loading real MaleCNS graph and checking fingerprints...", flush=True)
                    graph_cache["graph"] = load_lulzx_bundle(args.data_dir)
                brain = LiveConnectomeBrain(graph_cache["graph"], seed=seed)
                brain.template.advance(.1)
                brain.template.reset()
                return brain
        def dataset():
            return graph_cache["graph"].metadata if "graph" in graph_cache else {
                "name": "MaleCNS · loads on selection", "real_connectome": True}
        session = maze_session(seed=args.seed, connectome_factory=connectome_factory,
                               connectome_dataset=dataset, warmup_s=args.warmup_s)
        if args.controller == "connectome":
            if not args.data_dir:
                parser.error("--controller connectome requires --data-dir")
            session.control({"controller": "connectome"})
    server = create_server(session, host=args.host, port=args.port)
    print(f"Live demo: http://{args.host}:{server.server_port}/", flush=True)
    print(session.notice, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
