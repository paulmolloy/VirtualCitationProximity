import sys
import xml.etree.ElementTree as etree
import codecs
import csv
import time
import os
import pandas as pd

WIKI_ARTICLE_TITLES = '/media/pablo/large-fyp/articles_list_no_meta_list_insensitive.txt'
WIKI_ARTICLE_TITLES_SAMPLE = '/media/pablo/large-fyp/titles_sample_50000_no_lists.txt'
ENC = 'utf-8'
# sample gets a sample of N article titles to be used elsewhere.
def main():

    sample_size = 50000
    if len(sys.argv) == 2:
        sample_size = int(sys.argv[1])


    df = pd.read_csv(WIKI_ARTICLE_TITLES, sep='|', header=None)
    df.columns = ["title"]
    sample = df.sample(sample_size)
    print(sample)
    sample.to_csv(WIKI_ARTICLE_TITLES_SAMPLE, sep='|', encoding='utf-8', index=False)


if __name__ == '__main__':
    main()

