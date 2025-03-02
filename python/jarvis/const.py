"""
This module contains constants and default values used throughout the project. 
"""


from pathlib import Path
#! defines the project root directory (as the root of the gh repo) 
GHROOT = Path(__file__).parents[2]
#! if you move this file/folder, you need to change this line to match the new location. 
#! index matches the number of folders to go up from where THIS file is located: /[2] python/[1] jarvis/[0] const.py
# DEFAULT PATHS
DATADIR = GHROOT / 'datasets'
PYDIR = GHROOT / 'python'
PKGDIR = PYDIR / 'jarvis' 
KERNELDIR = "datasets/kernels/"
HST_DIR = DATADIR / 'HST'
HISAKI_DIR = DATADIR / 'Hisaki'
TORUS_DIR = HISAKI_DIR/'Torus Power'
AURORA_DIR = HISAKI_DIR/'Aurora Power'
# DEFAULT FITSINDEX: the default target HDU within a fits file.
FITSINDEX = 1 
# XY COORDS FOR IN DPR REGION OF IMAGES 
DPR_IMXY  = {'01':(584,1098), '02':(592,1150), '03':(742,1413), '04':(600,1233),
             '05':('',''), '06':(674,1233), '07':(607,1173), '08':(742,1390),
             '09':('',''), '10':(600,910), '11':(622,1061), '12':(584,1046), 
             '13':(570,1083), '14':(614,1241), '15':(660,1309), '16':(664,1264), 
             '17':(750,1391), '18':(614,971), '19':(607,1159), '20':('','')}
DPR_JSON = DATADIR / 'HST' / 'custom' / 'boundarypaths.json'


