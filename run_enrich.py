#!/usr/bin/env python
"""Thin wrapper: `python run_enrich.py --input "data (1).csv" --output data_full.csv`.

Starts the local Transfermarkt API in the background (bootstrapping it on first
run), enriches, and stops the server cleanly.
"""

from capology_pipeline.cli import main

if __name__ == "__main__":
    main()
