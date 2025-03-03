
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from jarvis import  hst_fitsfile_paths, get_obs_interval, fpath, fits_from_glob, get_data_over_interval
from datetime import datetime
from astropy.io import fits



infile = fpath('2025-02-28_21-18-05.txt')
def plot_visits(df, quantity='PFlux',corrected=None,ret='showsavefig',unit=None): #corrected = False, True, None (remove negative values)
    df['EPOCH'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['color'] = df[quantity].apply(lambda x: 'r' if x < 0 else 'b')
    corr_df = df.copy()
    if corrected is True:
        corr_df[quantity] = df[quantity].where(df[quantity] > 0, 0)
    elif corrected is None:
        corr_df[quantity] = df[quantity].where(df[quantity] > 0, np.nan)
    visits = df['visit'].unique()
    fitsdirs = hst_fitsfile_paths()[:-1]
    fig=plt.figure(figsize=(19.2,10.8), dpi=70)

    # this generates the grid dimensions based on the number of visits and a maximum of 8 columns
    N = len(visits)
    J = 1
    while True:
        I = N//J
        if N % J == 0 and I <= 8:
            break
        J += 1

    print(f'J={J}, I={I}')
    gs = fig.add_gridspec(J+1,I,  height_ratios=[3,*[1 for _ in range(J)]])#hspace=0,wspace=0,
    main_ax = fig.add_subplot(gs[0, :])
    main_ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=2))
    main_ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m/%y'))
    main_ax.xaxis.set_minor_locator(mpl.dates.DayLocator(interval=1))
    axs = [[fig.add_subplot(gs[j+1, i],label=f'v{visits[i+I*j]}') for i in range(I)] for j in range(J)]        
    for f in fitsdirs:
        mind,maxd=get_obs_interval(f)
        mind = pd.to_datetime(mind)
        maxd = pd.to_datetime(maxd)
        # span over the interval
        main_ax.axvspan(mind,maxd,  alpha=0.5, color='#aaa5')
    ylim = [corr_df['PFlux'].min(), corr_df['PFlux'].max()]
    diff = ylim[1] - ylim[0]
    yl = [ylim[0] + diff/10, ylim[1] - diff/10]
    for visit in visits:
        dfa = corr_df.loc[df['visit'] == visit]
        mean_d = dfa['EPOCH'].mean()
        if dfa[quantity].mean() > corr_df[quantity].mean():
            main_ax.text(mean_d,yl[0], visit, fontsize=10, color='black', va='top', ha='right',rotation=90) 
        else:
            main_ax.text(mean_d, yl[1], visit, fontsize=10, color='black', va='bottom', ha='right',rotation=90)
    main_ax.scatter(corr_df['EPOCH'], corr_df[quantity], marker='x',s=5, c=corr_df['color'])
    main_ax.xaxis.tick_top()
    for j in range(J):
        for i in range(I):
            dfa = corr_df.loc[df['visit'] == visits[i+I*j]]
            axs[j][i].plot(dfa['EPOCH'], dfa[quantity], color='#77f', linewidth=0.5)
            axs[j][i].scatter(dfa['EPOCH'], dfa[quantity], marker='.',s=5, c=dfa['color'])
            axs[j][i].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
            idfint = df.where(df['visit'] == visits[i+I*j])
            dmax,dmin =[idfint['EPOCH'].max(),idfint['EPOCH'].min()]
            delta = (dmax-dmin)          
            axs[j][i].set_xlim([dmin-0.01*delta, dmax+0.01*delta])
            axs[j][i].xaxis.set_major_locator(mpl.dates.MinuteLocator(interval=int(np.floor((delta.total_seconds()/60)/3))))
            axs[j][i].xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
            axs[j][i].annotate(f'v{visits[i+I*j]}'
                               ,xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top',weight='bold',
        bbox=dict(facecolor='#0000', edgecolor='none', pad=3.0))
            axs[j][i].annotate(f'{dmin.strftime("%d/%m/%y")}',xy=(1, 1), xycoords='axes fraction',
        xytext=(-0.05, +0.1), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='bottom',horizontalalignment='right', annotation_clip=False,
        bbox=dict(facecolor='#0000', edgecolor='none', pad=3.0))
            axs[j][i].tick_params(axis='both',pad=0)
    # for i in range(I):
    #     axs[0][i].xaxis.tick_top()
    #     axs[0][i].xaxis.set_label_position('top')
    for j in range(J):
        axs[j][0].set_ylabel(f'{quantity}'+(f' [{unit}]' if unit else ''))
    main_ax.set_ylabel(f'{quantity}'+(f' [{unit}]' if unit else ''))
    fig.suptitle(f'{quantity} over visits {df['visit'].min()} to {df['visit'].max()}', fontsize=20)
    if 'save' in ret:
        plt.savefig(fpath(f'figures/imgs/{quantity}_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.png'))
    if 'show' in ret:
        plt.show()
    if 'fig' in ret:
        return fig
print(infile)
df=pd.read_csv(infile, sep= ' ',index_col=False, names=['visit', 'Date', 'Time', 'Power', 'PFlux', 'Area'])
plot_visits(df, 'PFlux', unit='GW/km²')
plot_visits(df, 'Power', unit='GW')
plot_visits(df, 'Area', unit='km²')

testfits = fits.open(fpath('datasets/HST/group_13/jup_16-148-17-19-53_0100_v16_stis_f25srf2_proj.fits'))
table = get_data_over_interval(fits_from_glob(fpath("datasets/Hisaki/Torus Power/")), [datetime(2015,1,1), datetime(2017,1,1)], )
df = table.to_pandas()
df.plot(x='EPOCH',y='TPOW0710ADAWN')
plt.show()