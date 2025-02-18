import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
GHROOT = Path(__file__).parents[1]
def fpath(x):
    return os.path.join(GHROOT, x)

pd.options.display.max_rows = 9999

filename = r"datasets\Solar_Wind.txt"
table = pd.read_csv(fpath(filename), header=0,sep=r'\s+')
print(table.columns)

table['Time_UT'] = pd.to_datetime(table['Time_UT'], format='%Y-%m-%d-%H:%M:%S.%f') # see https://strftime.org/ for format codes
# plot the table:
table.plot(x='Time_UT', y='jup_sw_pdyn', title='Solar wind pressure vs Time at Jupiter', xlabel='Time_UT', ylabel='jup_sw_pdyn')

plt.title("Variation in solar wind pressure dynamics at Jupiter over 21 visits")
plt.xlabel("Date (UT)")
plt.ylabel("Solar wind pressure dynamics at Jupiter")
plt.show()  