import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

CPI_PAIRS_PATH = '/home/pablo/FYP/1000_with_titles_siamese_200_tokenized_results/cpi_tokenized_paragraphs_1000_sample_count_5.csv'
column_names = ['id', 'title_a', 'title_b', 'dist', 'count', 'id_a', 'id_b', 'cpi', 'avg_cpi', 'cpi_count_pt4', 'tokenized_title_a', 'tokenized_text_a', 'tokenized_title_b', 'tokenized_text_b']

cpi_pairs =  pd.read_csv(CPI_PAIRS_PATH, sep='|', header = None, names = column_names)
print(cpi_pairs.head())
fig, ax = plt.subplots()
ax.hist(cpi_pairs['cpi'], color = 'blue', edgecolor = 'black', bins = 200, log=True)
ax.set_title('Histogram of CPI Sum values of Citation Pairs where Count is Greater than 5')
plt.xlabel('CPI Value')
plt.ylabel('Num Citation Pairs (log)')



plt.show()


fig, ax = plt.subplots()
ax.hist(cpi_pairs['avg_cpi'], color = 'blue', edgecolor = 'black', bins = 200, log=True)
ax.set_title('Histogram of Average CPI values of Citation Pairs where Count is Greater than 5')
plt.xlabel('Average CPI Value')
plt.ylabel('Num Citation Pairs (log)')

plt.show()


fig, ax = plt.subplots()
ax.hist(cpi_pairs['cpi_count_pt4'], color = 'blue', edgecolor = 'black', bins = 200, log=True)
ax.set_title('Histogram of CPI/count^.4 for Citation Pairs where Count is Greater than 5')
plt.xlabel('CPI/count^.4')
plt.ylabel('Num Citation Pairs (log)')

plt.show()


