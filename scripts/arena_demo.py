"""Run the local behavioral arena lab, separately from the connectome/maze demo."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from flybrainflow.arena_server import ArenaSession, create_arena_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", choices=("open", "blocked", "surge"), default="open")
    args = parser.parse_args()
    try:
        session = ArenaSession({"population": args.population, "seed": args.seed, "preset": args.scenario})
        server = create_arena_server(session, port=args.port)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Arena lab: http://127.0.0.1:{server.server_port}/", flush=True)
    print("Experimental behavioral crowd model. Not connectome-controlled or validated for human safety.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
