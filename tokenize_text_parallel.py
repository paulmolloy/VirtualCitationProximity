import xml.etree.ElementTree as etree
import codecs
import csv
import time 
import os
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
import mwparserfromhell
from nltk import sent_tokenize 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import multiprocessing as mp


#  This takes in a csv of article titles and text and cleans up and tokenizes it down to 200 tokens
# in parallel.
WIKI_PATH = '/media/pablo/elements/'
ARTICLE_TEXTS = 'title_pages_full_parallelized_improved.csv'
TEST_ARTICLE_PATH = '/home/pablo/FYP/test/title_paragraph_untokenized.csv'
FYP_PATH = '/home/pablo/FYP/'
ARTICLES_TEXTS_PATH = os.path.join(WIKI_PATH, ARTICLE_TEXTS)
TOKENIZED_DEST_PATH = os.path.join(WIKI_PATH, 'tokenized_title_pages_full_parallelized.csv')
ENC = 'utf-8'
MAX_NUM_WORDS = 20000
MAX_QUEUE_SIZE = 20

count = 0
def tokenize(article):
    tokens = word_tokenize(article)
    # Get alpha numeric tokens
    words = [word.lower() for word in tokens if word.isalnum()]
    stop_words = stopwords.words('english')
    # Filter out stop words.
    words_no_stop = [w for w in words if not w in stop_words]
    # e.g. to keep a for the wiki article 'A'
    if len(words_no_stop) > 0:
        words = words_no_stop

    return words 

def process_page(pages, q):
    for p in pages:
        title = p[0]
        text = p[1]
        tokenized_text = tokenize(text)
        tokenized_title = tokenize(title)
        # print('Title: ' + title)
        # print('Len: ' + str(len(tokenized_text)))
        # print('Len title: ' + str(len(tokenized_title)))

        # Truncate to first 200 tokens.
        if len(tokenized_text) >= 200:
            tokenized_text = tokenized_text[:200]
        tokenized_title = tokenize(title)
        # Seems to be mainly Date articles. New text cleaning method will over come this.
        # Sql join will remove these pairs from the sample anyway. 
        if len(tokenized_text) > 0:
            line = title + '|' + ' '.join(tokenized_title) + '|' + ' '.join(tokenized_text) + '\n'
            q.put(line)

def line_writer_listener(q):
    # listens on q nd then writes to file.
    with open(TOKENIZED_DEST_PATH, 'w') as tokenized_file:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            tokenized_file.write(str(m))
            tokenized_file.flush()

debug = False

MAX_NUM_WORDS = 20000
PAGES_PER_THREAD = 50
manager = mp.Manager()
q = manager.Queue()
pool = mp.Pool(mp.cpu_count() + 2)
print('CPUs: ' + str(mp.cpu_count()))
watcher = pool.apply_async(line_writer_listener, (q,))
els = []

if debug:
    with open(TEST_ARTICLE_PATH, 'r') as paragraphs_file:
        for line in paragraphs_file:
            title_paragraph = line.split('|')

            # cleaned = clean_article()
            tokenized = tokenize(title_paragraph[1])
            print(tokenized)
            # Truncate to first 200 tokens.
            if len(tokenized) >= 200:
                tokenized = tokenized[:200]


else:
    with open(ARTICLES_TEXTS_PATH, 'r') as articles_text_file:
            for line in articles_text_file: 
                    title_paragraph = line.split('|')
                    text = title_paragraph[1]
                    title = title_paragraph[0]
                    els.append((title, text))
                    if len(els) >= PAGES_PER_THREAD:
                        pool.apply_async(process_page, args=(els, q))
                        while pool._taskqueue.qsize() > MAX_QUEUE_SIZE:
                            time.sleep(1)
                        els = []
            if len(els) != 0:
                job = pool.apply_async(process_page, args=(els, q))
            q.put('kill')
            pool.close()
            pool.join()

                    
