#################################################################
#                            KERNELEXTRACT
#################################################################
# This script is used to compress and decompress kernel files.
# command example:
#  >>>python python/kernelextract.py decompress
#  decompresses the kernel files from archive chunks
#  >>>python python/kernelextract.py compress
#  compresses the kernel files into archive chunks, not needed for normal operation
#  
#  extra options:
#  --kernel-dir: directory containing kernel files
#  --archive-dir: directory to store compressed chunks
#  --chunksize-mb: maximum chunk size in MB
#  --force: overwrite existing files
#  all options have default values though
#  @author: will-roscoe

import os
import tarfile
from tqdm import tqdm
from jarvis.utils import fpath
import argparse
if __name__ == "__main__":
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

    ARCHIVE_DIR = fpath('datasets/kernels/archive') if args.archive_dir == '.' else fpath(args.archive_dir)
    KERNEL_DIR = fpath('python/Kernels') if args.kernel_dir == '.' else fpath(args.kernel_dir)
    KERNELFILES = getattr(args,'files', ['de430.bsp', 'jup365.bsp', 'mk00062a.tsc', 'naif0012.tls', 'pck00011.tpc'])
    CHUNKSIZE_BYTES = getattr(args, 'chunksize_mb', 100) * 1024 * 1024
    DECOMPRESS_OVERWRITE = getattr(args, 'force', False)  # Use getattr to provide a default value

    if args.command == "compress":
        tqdm.write("KERNELEXTRACT: compressing kernel files with the following settings:")
        tqdm.write(f"ARCHIVE_DIR: {ARCHIVE_DIR}")
        tqdm.write(f"KERNEL_DIR: {KERNEL_DIR}")
        tqdm.write(f"CHUNKSIZE_BYTES: {CHUNKSIZE_BYTES}")
        tqdm.write(f"KERNELFILES: {KERNELFILES}")
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        tar_path = os.path.join(ARCHIVE_DIR, 'archive.tar')
        with tarfile.open(tar_path, 'w') as tar:
            for file in KERNELFILES:
                file_path = os.path.join(KERNEL_DIR, file)
                if os.path.isfile(file_path):
                    tar.add(file_path, arcname=file)
        
        with open(tar_path, 'rb') as f:
            total_size = os.path.getsize(tar_path)
            part_num = 1
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Compressing") as pbar:
                while chunk := f.read(CHUNKSIZE_BYTES):
                    part_path = os.path.join(ARCHIVE_DIR, f'archive.part{part_num}')
                    with open(part_path, 'wb') as part_file:
                        part_file.write(chunk)
                    part_num += 1
                    pbar.update(len(chunk))
        
        os.remove(tar_path)
        tqdm.write(f"Compression complete. {part_num - 1} parts created.")
    elif args.command == "decompress":
        tqdm.write("KERNELEXTRACT: decompressing kernel files with the following settings:")
        tqdm.write(f"ARCHIVE_DIR: {ARCHIVE_DIR}")
        tqdm.write(f"KERNEL_DIR: {KERNEL_DIR}")
        tqdm.write(f"OVERWRITE: {DECOMPRESS_OVERWRITE}")
        os.makedirs(KERNEL_DIR, exist_ok=True)
        tar_path = os.path.join(ARCHIVE_DIR, 'archive.tar')
        part_num = 1
        part_files = []
        while True:
            part_path = os.path.join(ARCHIVE_DIR, f'archive.part{part_num}')
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
                file_path = os.path.join(KERNEL_DIR, member.name)
                if os.path.exists(file_path) and not DECOMPRESS_OVERWRITE:
                    tqdm.write(f"Skipping {member.name}, already exists.")
                    continue
                tar.extract(member, path=KERNEL_DIR)
        
        os.remove(tar_path)
        tqdm.write("Decompression complete.")
    elif args.command == "cleanup":
        tqdm.write("KERNELEXTRACT: cleaning up decompressed files.")
        with tqdm(total=len(KERNELFILES), desc="Cleaning up") as pbar:
            for file in KERNELFILES:
                file_path = os.path.join(KERNEL_DIR, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                pbar.update(1)
        tqdm.write("Kernel files removed, only archive remaining.")
    else:
        raise ValueError("Invalid command. use --help for usage.")