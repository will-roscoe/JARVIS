#!/usr/bin/env python3
"""generate.py
Generate GIFs, images and Gaussian fits from FITS files
"""

import argparse
import os

from jarvis.utils import fits_from_glob, fpath
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass Figure Generation")
    subp = parser.add_subparsers(dest="command", required=True)
    gifgen = subp.add_parser("gifs", help="Generate GIFs from FITS files")
    imgen = subp.add_parser("imgs", help="Generate images from FITS files")
    gaussgen = subp.add_parser("gaussians", help="Generate Gaussian fits from FITS files")
    parser.add_argument(
        "--dir", type=str, nargs="+", required=False, default="NONE", help="directory to search for fits files",
    )
    # pass in as foo=bar x=y

    kwargs = {
        "region": {"type": bool, "help": "whether to plot regions"},
        "moonfp": {"type": bool, "help": "whether to plot moon footprints"},
        "fixed": {"type": bool, "help": "whether to plot fixed"},
        "full": {"type": bool, "help": "whether to plot full"},
        "cmap": {"type": str, "help": "colormap to use"},
        "norm": {"type": str, "help": "norm to use"},
        "dpi": {"type": int, "help": "dots per inch"},
        "fps": {"type": int, "default": 5, "help": "frames per second"},
        "crop": {"type": float, "help": "whether to crop the image"},
        "rlim": {"type": int, "help": "radius limit"},
    }
    for k, v in kwargs.items():
        gifgen.add_argument(f"--{k}", **v, required=False)
        imgen.add_argument(f"--{k}", **v, required=False)
        gaussgen.add_argument(f"--{k}", **v, required=False)

    args = parser.parse_args()

    if args.command == "gifs":
        from python.jarvis.plotting import make_gif
        from jarvis.utils import ensure_dir, filename_from_hdul

        ddir = fpath("datasets/HST")
        # find all directories in the HST directory, as absolute paths
        dirs = (
            [f for f in os.listdir(ddir) if os.path.isdir(os.path.join(ddir, f))]
            if args.dir == "NONE"
            else [fpath(a) for a in args.dir]
        )
        pb_ = tqdm(total=len(dirs), desc="Generating GIFs", unit="gifs")
        for d in dirs:
            make_gif(f"{ddir}/{d}", savelocation=fpath("figures/gifs/"), fps=args.fps)
            pb_.update(1)

    elif args.command == "imgs":
        from python.jarvis.plotting import moind
        from jarvis.utils import ensure_dir, filename_from_hdul
        from matplotlib import pyplot as plt

        ddir = fpath("datasets/HST")
        # find all directories in the HST directory, as absolute paths
        files_dir = (
            [f for f in os.listdir(ddir) if os.path.isdir(os.path.join(ddir, f))] if args.dir == "NONE" else [args.dir]
        )
        print(f"found {len(files_dir)} directories: {files_dir}")
        obs = []
        for d in files_dir:
            obs.append(fits_from_glob(f"{ddir}/{d}"))
        flat = [item for sublist in obs for item in sublist]
        ensure_dir(fpath("figures/imgs"))
        pbar = tqdm(total=len(flat), desc="Generating Images", unit="fr")
        for i, d in enumerate(obs):
            ensure_dir(fpath(f"figures/imgs/{files_dir[i]}"))
            pbar.set_postfix(file=files_dir[i])
            for f in d:
                fig, ax, fo = moind(f, **{k: v for k, v in vars(args).items() if v is not None})
                plt.savefig(fpath(f"figures/imgs/{files_dir[i]}/{filename_from_hdul(fo)}.png"))
                pbar.update(1)
                plt.close(fig)

    elif args.command == "gaussians":
        from jarvis.cvis import generate_coadded_fits
        from jarvis.utils import hst_fpath_list

        fpaths = hst_fpath_list()
        for i in tqdm(fpaths):
            fitsg = fits_from_glob(i)
            copath = fpath(f"datasets/HST/custom/{filename_from_hdul(i)}_coadded_gaussian.fits")
            fit = generate_coadded_fits(
                fitsg, saveto=copath, kernel_params=(3, 1), overwrite=False, indiv=False, coadded=True,
            )
            fit.close()
