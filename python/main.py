#! /usr/bin/env python3
"""Main script for running power calculations on Jupiter data."""
from jarvis.const import Presets
from jarvis.gen import run_path_powercalc
from jarvis.utils import fpath

if (__name__ == "__main__"):
    run_path_powercalc(
        ignores={"groups": [], "visits": []},
        byvisit=True,
        segments=1,
        coadd_dir=fpath("temp/coadds"),
        avg_dir=fpath("temp/rollavgs"),
        gifdir=fpath("temp/gifs"),
        outfile="auto",
        extname="BOUNDARY",
        remove="none",
        window=5,
        config = Presets.figure,
        fig_config = {"auto":True},
    )