from jarvis.utils import fpath, fits_from_glob
import argparse
import os
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass Figure Generation")
    subp = parser.add_subparsers(dest="command", required=True)
    gifgen = subp.add_parser("gifs", help="Generate GIFs from FITS files")
    imgen = subp.add_parser("imgs", help="Generate images from FITS files")
    gaussgen = subp.add_parser("gaussians", help="Generate Gaussian fits from FITS files")
    parser.add_argument("--dir", type=str, nargs='+',required=False, default='NONE', help="directory to search for fits files")
    # pass in as foo=bar x=y

    kwargs = dict(
        region=dict(type=bool,  help="whether to plot regions"),
        moonfp=dict(type=bool,  help="whether to plot moon footprints"),
        fixed=dict(type=bool,help="whether to plot fixed"),
        full=dict(type=bool,  help="whether to plot full"),
        cmap = dict(type=str, help="colormap to use"),
        norm = dict(type=str, help="norm to use"),
        dpi = dict(type=int,  help="dots per inch"),
        fps = dict(type=int, default=5, help="frames per second"),
        crop = dict(type=float,  help="whether to crop the image"),
        rlim = dict(type=int,  help="radius limit"),
    )
    for k,v in kwargs.items():
        gifgen.add_argument(f"--{k}", **v, required=False)
        imgen.add_argument(f"--{k}", **v, required=False)
        gaussgen.add_argument(f"--{k}", **v, required=False)



    args = parser.parse_args()

    if args.command == "gifs":
        from jarvis.polar import make_gif
        from jarvis.utils import make_filename, ensure_dir
        ddir = fpath('datasets/HST')
        #find all directories in the HST directory, as absolute paths
        dirs = [f for f in os.listdir(ddir) if os.path.isdir(os.path.join(ddir, f))] if args.dir == 'NONE' else [fpath(a) for a in args.dir]
        pbar = tqdm(total=len(dirs))
        for d in dirs:
            make_gif(f'{ddir}/{d}', savelocation=fpath("figures/gifs/"), fps=args.fps, )
            pbar.update(1)

    elif args.command == "imgs":
        from jarvis.polar import moind
        from matplotlib import pyplot as plt
        from jarvis.utils import make_filename, ensure_dir
        ddir = fpath('datasets/HST')
        #find all directories in the HST directory, as absolute paths
        files_dir = [f for f in os.listdir(ddir) if os.path.isdir(os.path.join(ddir, f))] if args.dir == 'NONE' else [args.dir]
        print(f"found {len(files_dir)} directories: {files_dir}")
        obs = []
        for d in files_dir:
            obs.append(fits_from_glob(f'{ddir}/{d}'))
        flat = [item for sublist in obs for item in sublist]
        ensure_dir(fpath("figures/imgs"))
        pbar = tqdm(total=len(flat), desc="Generating Images", unit="fr")
        for i,d in enumerate(obs):
            ensure_dir(fpath(f"figures/imgs/{files_dir[i]}"))
            pbar.set_postfix(file=files_dir[i])
            for f in d:
                fig,ax,fo = moind(f, **{k:v for k,v in vars(args).items() if v is not None})
                plt.savefig(fpath(f"figures/imgs/{files_dir[i]}/{make_filename(fo)}.png"))
                pbar.update(1)
                plt.close(fig)
         
    elif args.command == "gaussians":
        from jarvis.cvis import gen_gaussian_coadded_fits
        gen_gaussian_coadded_fits()
    