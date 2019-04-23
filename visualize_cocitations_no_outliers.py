import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

column_names = ['title', 'count']
title_counts =  pd.read_csv('/home/pablo/FYP/titles_sample1000_counts.csv', sep='|', header = None, names = column_names)
print(title_counts.head())
fig, ax = plt.subplots()
title_counts = title_counts[title_counts['count'] < 20000]
ax.hist(title_counts['count'], color = 'blue', edgecolor = 'black', bins = 200, log=True)
ax.set_title('Histogram of Number of Co-Citation Pairs Each of the Sampled 1000 Articles are in.')
plt.xlabel('Num Co-Citations')
plt.ylabel('Num Articles (log)')



plt.show()

