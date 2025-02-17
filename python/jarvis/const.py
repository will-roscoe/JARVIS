from pathlib import Path
#! defines the project root directory (as the root of the gh repo) 
GHROOT = Path(__file__).parents[2]
#! if you move this file/folder, you need to change this line to match the new location. 
#! index matches the number of folders to go up from where THIS file is located: /[2] python/[1] jarvis/[0] const.py

DATADIR = GHROOT / 'datasets'
PYDIR = GHROOT / 'python'
PKGDIR = PYDIR / 'jarvis' 
FITSINDEX = 1 #defines the default target HDU within a fits file.

# plot_defaults = dict(
# figure = dict(figsize=(7,6)),
# moonfp = dict(
#     colkey=lambda i: (('gold','IO'), ('aquamarine','EUR'), ('w','GAN'))[i],
#     text= dict(fontsize=10,alpha=0.5,path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],horizontalalignment='center', verticalalignment='center', fontweight='bold'),
#     p1 = dict(color='k', linestyle='-', lw=4),
#     p2 = dict( linestyle='-', lw=2.5)),
# cml = dict(plot=dict(color='r', linestyle='--', lw=1.2), text=dict(fontsize=11, color='r', horizontalalignment='center', verticalalignment='center', fontweight='bold')),



# )

