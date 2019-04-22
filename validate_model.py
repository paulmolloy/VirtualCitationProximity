# coding: utf-8
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import json
from wordcloud import WordCloud
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation 
from keras.layers.merge import concatenate
import keras.backend as K
from sklearn.model_selection import train_test_split
import keras.models

# validate_model loads each of the trained models and evaluates them against
# the test data-set. It displays a bar chart of the MAE of the models.

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', dest='debug', help= 'For debugging script use test file.', action='store_true')
parser.add_argument('-e', '--reload_embeddings', dest='reload_embeddings', help= 'Reload embeddings from file.', action='store_true')
parser.set_defaults(debug=False)
args = parser.parse_args()
MAX_NUM_WORDS= 50000
MAX_SEQUENCE_LENGTH = 200 # each title could be 100 tokens each.
GLOVE_PATH = '/home/pablo/FYP/glove.6B/glove.6B.100d.txt'
CPI_DATASET_DIR = '/home/pablo/FYP/1000_with_titles_siamese_2x_200_tokenized_results/'
CPI_DATASET = 'cpi_tokenized_paragraphs_1000_sample_count_5'
CPI_DATASET_FILE = CPI_DATASET + '.csv' 
CPI_DATASET_PATH = os.path.join(CPI_DATASET_DIR, CPI_DATASET_FILE)
CPI_BASE_DIR = '/home/pablo/FYP/'

TEST_DIR = '/home/pablo/FYP/test'

TINY_TEST_FILE = 'cpi_tokenized_200_test.csv'
NO_AVG_TEST_FILE = 'test_cpi_tokenized_200_1000_sample.csv'
TEST_FILE = 'cpi_tokenized_paragraphs_1000_sample_count_5.csv'
TEST_DATASET_PATH = os.path.join(TEST_DIR, TEST_FILE)

def rmse (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred -y_true), axis=-1))

#directory = CPI_DATASET_DIR
dataset_path = CPI_DATASET_PATH
# Columns titled "0" "199" are the emdeddings for  text_a
a_columns = [str(x) for x in range(200)]
# Columns titled "200" to "399" are the embeddings for text_b
b_columns = [str(x) for x in range(200, 400)]

# For the model that takes 50 words from each.
# Columns titled "0" "199" are the emdeddings for  text_a
a_columns_50 = [str(x) for x in range(50)]
# Columns titled "200" to "399" are the embeddings for text_b
b_columns_50 = [str(x) for x in range(50, 100)]

# read x_train back in from file:
column_names = ['title_a', 'title_b', 'texts_embeddings']

# directory-> run title
model_runs = {'1000_with_titles_convlstm2x_redo_200_tokenized_results': 'Baseline 1DConvnet LSTM', 
        '1000_with_titles_siamese_200_tokenized_results': 'Siamese LSTM',
        '1000_with_titles_siamese_2x_200_tokenized_results': 'Siamese 1DConvnet LSTM',
        '1000_with_titles_siamese_2x_no_dropout_200_tokenized_results': 'Siamese 1DConvnet LSTM No Dropout',
        '1000_with_titles_siamese_2x_50_tokenized_results': 'Siamese 1DConvnet LSTM Len 50'
        }
test_maes = []
print(model_runs)
for directory, title in model_runs.items():
    full_dir = os.path.join(CPI_BASE_DIR, directory)
    print(full_dir)
    x_test = pd.read_csv(os.path.join(full_dir, 'x_holdout_' + CPI_DATASET_FILE), sep='|', index_col='id')
    y_test = pd.read_csv(os.path.join(full_dir, 'y_holdout_' + CPI_DATASET_FILE), sep='|', index_col='id')
    # read back model
    model = keras.models.load_model(os.path.join(full_dir, 'saved-model-'+CPI_DATASET), custom_objects={'rmse': rmse})
    pred = []
    scores = []
    if title == 'Baseline 1DConvnet LSTM':
        pred = model.predict(x_test.drop(['title_a', 'title_b'], axis=1))
        scores = model.evaluate(x_test.drop(['title_a', 'title_b'], axis=1), y_test, verbose=0)
    elif title == 'Siamese 1DConvnet LSTM Len 50':
        pred = model.predict([x_test[a_columns_50], x_test[b_columns_50]])
        scores = model.evaluate([x_test[a_columns_50], x_test[b_columns_50]], y_test, verbose=0)
    else:
        pred = model.predict([x_test[a_columns], x_test[b_columns]])
        scores = model.evaluate([x_test[a_columns], x_test[b_columns]], y_test, verbose=0)

    results = pd.concat([x_test['title_a'], x_test['title_b'] , y_test], axis=1, join_axes=[y_test.index])
    results['pred_avg_cpi'] = pred
    results.to_csv(os.path.join(full_dir, 'final_results_conv_glove_100d_' + CPI_DATASET_FILE), sep='|', index=False)
    print("%s: %.6f" % (model.metrics_names[1], scores[1]))
    test_maes.append(scores[1])

print(test_maes)
y_pos = np.arange(len(model_runs))
model_titles = list(model_runs.values())
fig, ax = plt.subplots()
plt.bar(y_pos, test_maes, align='center', alpha=0.5)
plt.xticks(y_pos, model_titles)
ax.set_title('MAE Results After 5 Epochs trained on 176841 CPI Pairs for 1k Articles')
plt.xlabel('Model')
plt.ylabel('Mean Average Error')



plt.show()

