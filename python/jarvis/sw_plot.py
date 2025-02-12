import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from const import fpath

#Time = np.arange(1, 13)
#Intensity = (8,7,5,3,4,12,9,4,17,12,11,5)

#ypoints = Intensity
#xpoints = Time

#plt.plot(xpoints, ypoints)
#plt.xlabel("Time")
#plt.ylabel("Intensity")

#plt.show()

filename = "datasets\Solar_Wind.txt"
table = pd.read_csv(fpath(filename), sep=r'\s+', comment='#')

times = table['Time_UT']
values = table['jup_sw_pdyn']
# access a row of the table using the row index:
row = table.iloc[0]
# remove rows with empty values:
table = table.dropna()
# or fill empty values with a default value:
table = table.fillna(0)
# get statistics of the table:
stats = table.describe()
# get the row indexes with values greater than 10:
indexes = table[table['jup_sw_pdyn'] > 10].index
# or similarly get the rows with values greater than 10:
rows = table[table['jup_sw_pdyn'] > 10]
# turn a datetime collumn into a datetime object, which can be used for plotting:
table['Time_UT'] = pd.to_datetime(table['Time_UT'], format='%Y-%m-%d_%H:%M:%S') # see https://strftime.org/ for format codes
# plot the table:
table.plot(x='Time_UT', y='jup_sw_pdyn', title='jup_sw_pdyn vs Time_UT', xlabel='Time_UT', ylabel='jup_sw_pdyn')