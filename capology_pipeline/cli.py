"""CLI entry point for the enrichment pipeline."""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .enricher import PlayerEnricher


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Enrich a players CSV with Capology salaries + Transfermarkt market value."
    )
    parser.add_argument("--input", default=None, help="input CSV (default: 'data (1).csv')")
    parser.add_argument("--output", default=None, help="output CSV (default: data_full.csv)")
    parser.add_argument("--tm-port", type=int, default=None, help="local Transfermarkt API port")
    parser.add_argument("--capology-workers", type=int, default=None, help="Capology thread workers")
    parser.add_argument("--chunk-size", type=int, default=None, help="enable chunk/resume mode with N rows per chunk")
    parser.add_argument("--chunks-dir", default=None, help="directory for chunk outputs (default: <output_stem>_chunks)")
    args = parser.parse_args(argv)

    config = Config()
    if args.tm_port is not None:
        config.tm_port = args.tm_port
    if args.capology_workers is not None:
        config.capology_workers = args.capology_workers

    try:
        with PlayerEnricher(config) as enricher:
            if args.chunk_size:
                enricher.run_chunked(args.input, args.output, args.chunk_size, args.chunks_dir)
            else:
                enricher.run(args.input, args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
