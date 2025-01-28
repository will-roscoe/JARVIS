from pathlib import Path
import os
# absolute path to the root of the project
GHROOT = Path(__file__).parents[1]
# takes a relative path within repo and returns an absolute path.
fpath = lambda x: os.path.join(GHROOT, x)

# test, prints the README at the root of the project
def __testpaths():
    for x in os.listdir(GHROOT):
        print(x)
