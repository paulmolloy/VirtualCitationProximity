import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

column_names = ['title', 'count']
title_counts =  pd.read_csv('/home/pablo/FYP/sample-analysis/titles_sample_1000/titles_sample1000_counts.csv', sep='|', header = None, names = column_names)
print(title_counts.head())
fig, ax = plt.subplots()
ax.hist(title_counts['count'], color = 'blue', edgecolor = 'black', bins = 200, log=True)
ax.set_title('Histogram: Number of Co-Citations for the 1000 Sample Articles.')
plt.xlabel('Num Co-Citations')
plt.ylabel('Num Articles (log)')



plt.show()

