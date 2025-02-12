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

filename = "Solar_Wind.txt"
table = pd.read_csv(fpath(filename), sep=r'\s+', comment='#')

times = table['Times']
values = table['Values']
# access a row of the table using the row index:
row = table.iloc[0]
# remove rows with empty values:
table = table.dropna()
# or fill empty values with a default value:
table = table.fillna(0)
# get statistics of the table:
stats = table.describe()
# get the row indexes with values greater than 10:
indexes = table[table['Values'] > 10].index
# or similarly get the rows with values greater than 10:
rows = table[table['Values'] > 10]
# turn a datetime collumn into a datetime object, which can be used for plotting:
table['Times'] = pd.to_datetime(table['Times'], format='%d-%m-%Y_%H:%M:%S') # see https://strftime.org/ for format codes
# plot the table:
table.plot(x='Times', y='Values', title='Values vs Time', xlabel='Time', ylabel='Values')