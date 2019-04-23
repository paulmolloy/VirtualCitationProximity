import sys
import xml.etree.ElementTree as etree
import codecs
import csv
import time
import os
import pandas as pd

WIKI_ARTICLE_TITLES = '/media/pablo/large-fyp/sample_10000_not_related_pairs.txt'
BALANCED_NOT_RELATED_CLASS_SAMPLE = '/media/pablo/large-fyp/sample_10000_balanced_not_related_pairs.txt'
ENC = 'utf-8'
# class_imbalance gets a random sample of sample_size cpi pairs.
def main():

    sample_size = 339101
    if len(sys.argv) == 2:
        sample_size = int(sys.argv[1])


    df = pd.read_csv(WIKI_ARTICLE_TITLES, sep='|', header=None)
    print(df.head())
    #df.columns = ['id', 'title_a', 'title_b', 'dist', 'count', 'id_a', 'id_b', 'cpi']

    sample = df.sample(sample_size)
    print(sample)
    sample.to_csv(BALANCED_NOT_RELATED_CLASS_SAMPLE, sep='|', encoding='utf-8', index=False)


if __name__ == '__main__':
    main()

