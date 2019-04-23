import xml.etree.ElementTree as etree
import codecs
import csv
import time 
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import mwparserfromhell
from nltk import sent_tokenize 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import multiprocessing as mp

WIKI_PATH = '/media/pablo/elements/'
WIKI_FILENAME = 'enwiki-20190101-pages-articles-multistream.xml'
TEST_ARTICLE_PATH = '/home/pablo/repos/FYP-VCP-workspace/example_text_wiles.txt'
FYP_PATH = '/home/pablo/FYP/'
ENC = 'utf-8'
MAX_NUM_WORDS = 20000
path_wiki_xml = os.path.join(WIKI_PATH, WIKI_FILENAME)
max_queue_size = 20
count = 0
# Wiki XML to title article text CSV in parallel.
def get_tag_name(t):
    idx  = t.rfind("}")
    if idx != -1:
        t = t[idx +1:]
    return t

def iterate_xml(xml_file):
    doc = etree.iterparse(xml_file, events=('start', 'end'))
    _, root = next(doc)
    start_tag = None
    for event, el in doc:
        cur_tag = get_tag_name(el.tag)
        if event == 'start' and start_tag is None:
            start_tag = cur_tag
        if event ==  'end' and cur_tag == start_tag:
            yield el
            start_tag = None
            root.clear()


# Given a string of wikipedia markeddown text
# cleans non-plaintext markup.
# This is not perfect as unescaped brackets in the
# text mess it up and some xml tags are miss matched
# in the wiki dump etc.
def clean_article(article):
    # Everything from see also. 
    result = article 
    result = re.sub(r'==(\ *See also\ *|\ *External links\ *|\ *References\ *|\ *Notes\ *)==(.*)', r"", 
            article, flags=re.DOTALL)
    wikicode = mwparserfromhell.parse(result)
    plain_text = wikicode.strip_code()

    # [[File|Image:something]]
    plain_text = re.sub(r"(thumb\|)(.*)", r"", plain_text)
    # lines starting with | or {
    plain_text = re.sub(r"[\{|\|](.*)", r" ", plain_text)
    plain_text = re.sub(r"(\||\n)", r" ", plain_text)
    # any pipes or new lines.

    plain_text = plain_text.replace('|', ' ')
    return plain_text.replace('\n', ' ')

def process_page(pages, paragraph_file):
    for el in pages:
        title = ''
        redirect = ''
        id = -1
        revision = None
        is_redirect = False
        for child in el:
            cur_tag = get_tag_name(child.tag)
            if cur_tag == "title":
                title = child.text
            elif cur_tag == 'id':
                id = int(child.text)
            elif cur_tag == 'redirect':
                redirect = child.attrib['title']
            elif cur_tag == 'revision':
                revision = child

        if( redirect == '' and not title.startswith('File:') and not title.startswith('Wikipedia:')
                and not title.startswith('Template:') and not title.startswith('Category:')
                and not title.startswith('Help:') and not title.startswith('Portal:')
                and not title.startswith('Draft:') and not title.startswith('Module:')
                and not title.startswith('Book:') and not title.startswith('MediaWiki:')):
            model = ""
            text = ""
            for child in revision:
                cur_tag = get_tag_name(child.tag)
                if cur_tag == "text":
                    text = child.text
                elif cur_tag == 'model':
                    model = child.text
            cleaned = clean_article(text)
            #tokenized_text = tokenize(cleaned)
            # Truncate to first 200 tokens.
            #if len(tokenized_text) >= 200:
            #    tokenized_text = tokenized_text[:200]
            #tokenized_title = tokenize(title)
            q.put(title + '|' + cleaned +'\n')

def line_writer_listener(q):
    # listens on q and the writes to file. 
    with open(WIKI_PATH+ 'title_pages_full_parallelized_improved.csv', 'w') as paragraph_file:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            paragraph_file.write(str(m))
            paragraph_file.flush()


debug = False
if debug:
    with open(TEST_ARTICLE_PATH, 'r') as myfile:
        data=myfile.read()
        cleaned = clean_article(data)
        tokenized = tokenize(cleaned)
        # Truncate to first 200 tokens.
        if len(tokenized) >= 200:
            tokenized = tokenized[:200]

else:
    MAX_NUM_WORDS = 20000
    PAGES_PER_THREAD = 50
    df = pd.DataFrame(columns=['title', 'paragraph'])
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)
    print(' Cpus: ' + str(mp.cpu_count()))
    watcher = pool.apply_async(line_writer_listener, (q,))
    els = []
    i = 0

    for el in iterate_xml(path_wiki_xml):
        # print el
        tname = get_tag_name(el.tag)
        jobs = []
        if tname == 'page':
            # parallelize here
            els.append(el)
            if(len(els)==PAGES_PER_THREAD):
                job = pool.apply_async(process_page, args=(els, q))
                jobs.append(job)
                while pool._taskqueue.qsize() > max_queue_size:
                    time.sleep(1)
                els = [] 
    if len(els) != 0:
        job = pool.apply_async(process_page, args=(els, q))
    for job in jobs:
       job.get()

    q.put('kill')
    pool.close()

