import os
import pandas as pd
import csv
pairs_dir = '/media/pablo/large-fyp/'
pairs_name = 'filtered_5_no-lists-insensitive-all-citation-pairs.csv'
pairs_path = os.path.join(pairs_dir + pairs_name)
os.path.getsize(pairs_path)

column_names = ['hash', 'title_a', 'title_b', 'dist', 'count', 'page_idA', 'page_idB', 'cpi']
df_sample = pd.read_csv(pairs_path,sep='|', nrows=10, names=column_names)
print(df_sample.head())
df_sample_size = df_sample.memory_usage(index=False).sum();

# chunk size of 1 gb of csv text.
Gb = 1000000000
chunk = (Gb/ df_sample_size)/10
chunk = int(chunk//1)

# Create iterator

iter_csv = pd.read_csv(
        pairs_path,
        iterator=True,
        chunksize=chunk,
        sep='|', 
        names=column_names)


with  open (pairs_dir+'related_sum_filtered_5_' + pairs_name, 'a') as f_related:
    with  open (pairs_dir+'not_related_sum_filtered_5_' + pairs_name, 'a') as f_not:
        for chunk in iter_csv:
            cpi = chunk['cpi']
            chunk[cpi>=1].to_csv(f_related, header=False, index=False, sep='|')
            chunk[cpi<1].to_csv(f_not, header=False, index=False, sep='|')



