import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from keras.layers import Embedding
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
MAX_NUM_WORDS= 200
MAX_SEQUENCE_LENGTH = 100


column_names = ['id', 'title_a', 'title_b', 'dist', 'count', 'id_a', 'id_b', 'cpi', 'tokenized_title_a', 'tokenized_text_a', 'tokenized_title_b', 'tokenized_text_b']
train = pd.read_csv('/home/pablo/FYP/sample-all-citation-pairs1000-no-id.csv', sep='|', header = None, names = column_names)
# 2|United States|Lily van den Broecke雲|13275|1|1919370|36828443|0.000298404574868282|United States|This is the article about usa|Lily van den Broecke雲| This person sounds雲 dutch yū.

title_a = train['tokenized_text_a']
title_b = train['tokenized_text_b']

with open('/home/pablo/FYP/titles_sample1000.txt') as titles_sample_list:
    with open('/home/pablo/FYP/titles_sample1000_counts.csv', 'w') as titles_count:
        for title in titles_sample_list:
            title = title.rstrip()
            num_cocitations = train['title_b'].str.count(title).sum() + train['title_a'].str.count(title).sum()
            count_line = title + '|' + str(num_cocitations) + '\n'
            print(count_line)
            titles_count.write(count_line)



'''
fig, ax = plt.subplots(1,2, figsize=(20,5))
ax[0].imshow(wordcloud)
ax[0].set_title('Top words in wiki articles a ', fontsize = 20)
ax[0].axis('off')

wordcloud = WordCloud(max_font_size=None, stopwords=None,
        scale = 2, colormap = 'Dark2').generate(train_str_b)

ax[1].imshow(wordcloud)
ax[1].set_title('Top words in wiki articles b', fontsize = 20)
ax[1].axis('off')

plt.show()
print('t_text_a: ' + train['tokenized_text_a'])
print('t_text_b: ' + train['tokenized_text_b'])
'''
