from pathlib import Path
import os
import datetime
#! defines the project root directory (as the root of the gh repo) 
GHROOT = Path(__file__).parents[2]
#! if you move this file/folder, you need to change this line to match the new location. 
#! index matches the number of folders to go up from where THIS file is located: /[2] python/[1] jarvis/[0] const.py
# takes a relative path within repo and returns an absolute path.
fpath = lambda x: os.path.join(GHROOT, x)



# test, prints the README at the root of the project
def __testpaths():
    for x in os.listdir(GHROOT):
        print(x)
        
