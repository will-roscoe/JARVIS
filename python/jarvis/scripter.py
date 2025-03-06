"""################################################################
                            KERNELEXTRACT
################################################################
 This script is used to compress and decompress kernel files.
 command example:
  >>>python ..../kernelextract.py decompress
  decompresses the kernel files from archive chunks
  >>>python ..../kernelextract.py compress
 compresses the kernel files into archive chunks, not needed for normal operation
 
 extra options:
 --kernel-dir: directory containing kernel files
 --archive-dir: directory to store compressed chunks
 --chunksize-mb: maximum chunk size in MB
 --force: overwrite existing files
 all options have default values though
 @author: will-roscoe"""

import os
import tarfile
from tqdm import tqdm
from jarvis.utils import fpath
from jarvis.const import KERNELDIR
import argparse

from jarvis.utils import  fits_from_glob
from jarvis.polar import make_gif
from jarvis.polar import moind
from matplotlib import pyplot as plt
from jarvis.utils import filename_from_hdul, ensure_dir

    

    
class ScriptLike:
    def __init__(self, args):
        self.args = args
    
class KernelExtract(ScriptLike):
    """################################################################
                            KERNELEXTRACT
################################################################
 This script is used to compress and decompress kernel files.
 command example:
```>>>python ..../kernelextract.py decompress```
  decompresses the kernel files from archive chunks
```>>>python ..../kernelextract.py compress```
 compresses the kernel files into archive chunks, not needed for normal operation
 
 as an object: we can use the object as a context manager to compress and decompress files
    ```>>>with KernelExtract() as ke:
    >>>    foo = bar() #do something with the kernel files
    >>> xyz = abc(foo) #do something else with the kernel files
    >>>    #the kernel files are compressed and then decompressed when the context manager exits```

 extra options:
 --kernel-dir: directory containing kernel files
 --archive-dir: directory to store compressed chunks
 --chunksize-mb: maximum chunk size in MB
 --force: overwrite existing files
 all options have default values though
 @author: will-roscoe"""
    def __init__(self, args=None):
        """Compress, decompress and cleanup kernel files."""
        if __name__ == "__main__" and args is None:
            parser = argparse.ArgumentParser(description="Compress, decompress and cleanup kernel files.")
            subparsers = parser.add_subparsers(dest="command", required=True)
            compress_parser = subparsers.add_parser("compress", help="Compress kernel files into chunks.")
            compress_parser.add_argument("--kernel-dir", type=str, required=False, default=".", help="Directory containing kernel files.")
            compress_parser.add_argument("--archive-dir", type=str, required=False, default=".", help="Directory to store compressed chunks.")
            compress_parser.add_argument("--chunksize-mb", type=int, required=False, default=100, help="Maximum chunk size in MB.")
            compress_parser.add_argument("--files", type=str, nargs='+', required=False, default=['de430.bsp', 'jup365.bsp', 'mk00062a.tsc', 'naif0012.tls', 'pck00011.tpc'], help="Files to compress.")
            decompress_parser = subparsers.add_parser("decompress", help="Decompress chunked archive.")
            decompress_parser.add_argument("--archive-dir", type=str, required=False, default=".", help="Directory containing compressed chunks.")
            decompress_parser.add_argument("--kernel-dir", type=str, required=False, default=".", help="Directory to extract files to.")
            decompress_parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
            cleanup_parser = subparsers.add_parser("cleanup", help="cleanup decompressed files")
            cleanup_parser.add_argument("--archive-dir", type=str, required=False, default=".", help="Directory containing compressed chunks.")
            cleanup_parser.add_argument("--kernel-dir", type=str, required=False, default=".", help="Directory to extract files to.")
            args = parser.parse_args()
            self.ARCHIVE_DIR = fpath(KERNELDIR+'archive') if args.archive_dir == '.' else fpath(args.archive_dir)
            self.KERNEL_DIR = fpath(KERNELDIR) if args.kernel_dir == '.' else fpath(args.kernel_dir)
            self.KERNELFILES = getattr(args,'files', ['de430.bsp', 'jup365.bsp', 'mk00062a.tsc', 'naif0012.tls', 'pck00011.tpc'])
            self.CHUNKSIZE_BYTES = getattr(args, 'chunksize_mb', 100) * 1024 * 1024
            self.DECOMPRESS_OVERWRITE = getattr(args, 'force', False)  # Use getattr to provide a default value
        if args is not None:
            if args.command == "compress":
                self.compress()
            elif args.command == "decompress":
                self.decompress()
            elif args.command == "cleanup":
                self.cleanup()
            else:
                raise ValueError("Invalid command. use --help for usage.")
            super().__init__(args)
    def compress(self,args=None):
        """Compress kernel files into chunks."""
        tqdm.write("KERNELEXTRACT: compressing kernel files with the following settings:")
        tqdm.write(f"ARCHIVE_DIR: {self.ARCHIVE_DIR}")
        tqdm.write(f"KERNEL_DIR: {self.KERNEL_DIR}")
        tqdm.write(f"CHUNKSIZE_BYTES: {self.CHUNKSIZE_BYTES}")
        tqdm.write(f"KERNELFILES: {self.KERNELFILES}")
        os.makedirs(self.ARCHIVE_DIR, exist_ok=True)
        tar_path = os.path.join(self.ARCHIVE_DIR, 'archive.tar')
        with tarfile.open(tar_path, 'w') as tar:
            for file in self.KERNELFILES:
                file_path = os.path.join(self.KERNEL_DIR, file)
                if os.path.isfile(file_path):
                    tar.add(file_path, arcname=file)
        
        with open(tar_path, 'rb') as f:
            total_size = os.path.getsize(tar_path)
            part_num = 1
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Compressing") as pbar:
                while chunk := f.read(self.CHUNKSIZE_BYTES):
                    part_path = os.path.join(self.ARCHIVE_DIR, f'archive.part{part_num}')
                    with open(part_path, 'wb') as part_file:
                        part_file.write(chunk)
                    part_num += 1
                    pbar.update(len(chunk))
        
        os.remove(tar_path)
        tqdm.write(f"Compression complete. {part_num - 1} parts created.")
    def decompress(self,args=None):
        """Decompress kernel files from archive chunks."""
        tqdm.write("KERNELEXTRACT: decompressing kernel files with the following settings:")
        tqdm.write(f"ARCHIVE_DIR: {self.ARCHIVE_DIR}")
        tqdm.write(f"KERNEL_DIR: {self.KERNEL_DIR}")
        tqdm.write(f"OVERWRITE: {self.DECOMPRESS_OVERWRITE}")
        os.makedirs(self.KERNEL_DIR, exist_ok=True)
        tar_path = os.path.join(self.ARCHIVE_DIR, 'archive.tar')
        part_num = 1
        part_files = []
        while True:
            part_path = os.path.join(self.ARCHIVE_DIR, f'archive.part{part_num}')
            if not os.path.exists(part_path):
                break
            part_files.append(part_path)
            part_num += 1
        
        total_size = sum(os.path.getsize(part) for part in part_files)
        
        with open(tar_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="Decompressing") as pbar:
            for part_path in part_files:
                with open(part_path, 'rb') as part_file:
                    chunk = part_file.read()
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                file_path = os.path.join(self.KERNEL_DIR, member.name)
                if os.path.exists(file_path) and not self.DECOMPRESS_OVERWRITE:
                    tqdm.write(f"Skipping {member.name}, already exists.")
                    continue
                tar.extract(member, path=self.KERNEL_DIR)
        
        os.remove(tar_path)
        tqdm.write("Decompression complete.")
    def cleanup(self,args=None):
        """Remove decompressed kernel files, leaving only the archive."""
        tqdm.write("KERNELEXTRACT: cleaning up decompressed files.")
        with tqdm(total=len(self.KERNELFILES), desc="Cleaning up") as pbar:
            for file in self.KERNELFILES:
                file_path = os.path.join(self.KERNEL_DIR, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                pbar.update(1)
        tqdm.write("Kernel files removed, only archive remaining.")
    def __enter__(self):
        self.compress()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        return False









 



class Generator(ScriptLike):
    """Generate GIFs, images and Gaussian fits from FITS files"""
    def __init__(self, args=None):
        """Generate GIFs, images and Gaussian fits from FITS files"""
        if __name__ == "__main__" and args is None:
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
        if args is not None:
            if args.command == "gifs":
                self.gifs()
            elif args.command == "imgs":
                self.imgs()
            elif args.command == "gaussians":
                self.gaussians()
            super().__init__(args)
    def gifs(self,args=None):
        """Generate GIFs from FITS files"""
        ddir = fpath('datasets/HST')
        #find all directories in the HST directory, as absolute paths
        dirs = [f for f in os.listdir(ddir) if os.path.isdir(os.path.join(ddir, f))] if args.dir == 'NONE' else [fpath(a) for a in args.dir]
        pb_ = tqdm(total=len(dirs), desc="Generating GIFs", unit="gifs")
        for d in dirs:
            make_gif(f'{ddir}/{d}', savelocation=fpath("figures/gifs/"), fps=args.fps, )
            pb_.update(1)
    def imgs(self,args=None):
        """Generate images from FITS files"""
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
                plt.savefig(fpath(f"figures/imgs/{files_dir[i]}/{filename_from_hdul(fo)}.png"))
                pbar.update(1)
                plt.close(fig)
    def gaussians(self,args=None):
        """Generate Gaussian fits from FITS files"""
        from jarvis.cvis import generate_coadded_fits
        from jarvis.utils import hst_fpath_list, filename_from_hdul
        fpaths = hst_fpath_list()
        for i in tqdm(fpaths):
            fitsg = fits_from_glob(i)
            copath = fpath(f'datasets/HST/custom/{filename_from_hdul(i)}_coadded_gaussian.fits')
            fit = generate_coadded_fits(fitsg, saveto=copath, kernel_params=(3,1), overwrite=False,indiv=False, coadded=True)
            fit.close()

class Tidy(ScriptLike):
    """UNTESTED"""
    def __init__(self,args=None):
        """WARNING: THIS SCRIPT WILL DELETE ALL FILES IN THE DIRECTORIES THAT ARE NOT IN THE NORMALDIRS LIST"""
        if __name__ == "__main__" and args is None:
            parser = argparse.ArgumentParser(description="Tidy up the HST data directories.")
            parser.add_argument("--dir", type=str, nargs='+',required=False, default='NONE', help="directory to search for fits files")
            args = parser.parse_args()
        if args is not None:
  
            
            super().__init__(args)
    








    
        
    