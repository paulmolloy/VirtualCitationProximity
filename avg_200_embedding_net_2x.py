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
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation 
import keras.backend as K
from sklearn.model_selection import train_test_split

# avg_200_embedding_siamese_2x_net preprocceses the data-set into word embeddings and trains the baseline sequential
# 1DCOnvent LSTM model on the training data.

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', dest='debug', help= 'For debugging script use test file.', action='store_true')
parser.add_argument('-e', '--reload_embeddings', dest='reload_embeddings', help= 'Reload embeddings from file.', action='store_true')

parser.set_defaults(debug=False)
args = parser.parse_args()
MAX_NUM_WORDS= 50000
MAX_SEQUENCE_LENGTH = 400 # each title could be 100 tokens each.
GLOVE_PATH = '/home/pablo/FYP/glove.6B/glove.6B.100d.txt'
CPI_DATASET_DIR = '/home/pablo/FYP/1000_with_titles_convlstm2x_redo_200_tokenized_results'
CPI_DATASET = 'cpi_tokenized_paragraphs_1000_sample_count_5'
CPI_DATASET_FILE = CPI_DATASET + '.csv' 
CPI_DATASET_PATH = os.path.join(CPI_DATASET_DIR, CPI_DATASET_FILE)

TEST_DIR = '/home/pablo/FYP/test'

TINY_TEST_FILE = 'cpi_tokenized_200_test.csv'
TEST_FILE = 'test_cpi_tokenized_200_1000_sample.csv'
TEST_DATASET_PATH = os.path.join(TEST_DIR, TEST_FILE)

def rmse (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred -y_true), axis=-1))

column_names = ['id', 'title_a', 'title_b', 'dist', 'count', 'id_a', 'id_b', 'cpi', 'tokenized_title_a', 'tokenized_text_a', 'tokenized_title_b', 'tokenized_text_b']
print('Debug = ' + str(args.debug))
if(args.debug):
    directory = TEST_DIR
    dataset_path = TEST_DATASET_PATH
else:
    directory = CPI_DATASET_DIR
    dataset_path = CPI_DATASET_PATH
if args.reload_embeddings:
    data = pd.read_csv(dataset_path, sep='|',  names = column_names)

    # Format example: 
    #
    # 2|United States|Lily van den Broecke雲|13275|1|1919370|36828443|0.000298404574868282|United States|This is the article about usa|Lily van den Broecke雲| This person sounds雲 dutch yū.

    cpi_problems = data[data['avg_cpi'] == np.nan]
    if len(cpi_problems) != 0:
            print('Some cpis are Nan clean data some more.')
    data = data.replace(np.nan, '', regex=True)
    # tokenized_titles_ab = data['tokenized_title_a'].map(str) + ' ' + data['tokenized_title_b']
    tokenized_texts_ab = data['tokenized_text_a'].map(str) + ' ' + data['tokenized_text_b']
    print(tokenized_texts_ab.head())

    # word cloud stuff
    '''
    wordcloud = WordCloud(max_font_size=None, stopwords=None,
            scale = 2, colormap = 'Dark2').generate(all_str_titel_a)

    fig, ax = plt.subplots(1,2, figsize=(20,5))
    ax[0].imshow(wordcloud)
    ax[0].set_title('Top words in wiki articles a ', fontsize = 20)
    ax[0].axis('off')

    wordcloud = WordCloud(max_font_size=None, stopwords=None,
            scale = 2, colormap = 'Dark2').generate(all_str_title_b)

    ax[1].imshow(wordcloud)
    ax[1].set_title('Top words in wiki articles b', fontsize = 20)
    ax[1].axis('off')

    plt.show()
    #print('t_text_a: ' + all_data['tokenized_text_a'])
    #print('t_text_b: ' + all_data['tokenized_text_b'])

    '''
    # load embeddings_index of {word: glove_coefficients}
    embeddings_index = {}
    with open(GLOVE_PATH) as f:
        for line in f:
            values = line.split(' ')
            word = values[0] # The word
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients

    print('Loaded GloVe embedding')
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    # Creates internal word index based on frequency 
    tokenizer.fit_on_texts(tokenized_texts_ab)
    del tokenized_texts_ab
    print('Finish fit on texts')

    # Transforms texts into sequence of integers based on the tokenizers 
    # word index.
    sequences_texts_a = tokenizer.texts_to_sequences(data['tokenized_text_a'].map(str))
    sequences_texts_b = tokenizer.texts_to_sequences(data['tokenized_text_b'].map(str))
    #sequences_b = tokenizer.texts_to_sequences(train['tokenized_title_b'])

    word_index = tokenizer.word_index
    print('Len word_idex:' + str(len(word_index)))
    if args.debug:
        print('Word index: ' + str(word_index))
    # add zeros to the vector to make it len 200
    p_a = pad_sequences(tokenizer.texts_to_sequences(data['tokenized_text_a'].map(str)), maxlen=int(MAX_SEQUENCE_LENGTH/2))
    p_b = pad_sequences(tokenizer.texts_to_sequences(data['tokenized_text_b'].map(str)), maxlen=int(MAX_SEQUENCE_LENGTH/2))
    texts_embeddings = pd.concat([pd.DataFrame(p_a), pd.DataFrame(p_b)], axis=1)
    texts_embeddings.columns = [x for x in range(400)]
    del p_a
    del p_b
    print('made embeddings frame')
    cpis = data['avg_cpi']
    cpis.index.rename('id',inplace=True)
    print('Got cpis etc')
    data = pd.concat( [data['title_a'], data['title_b'], texts_embeddings], axis=1, join_axes=[texts_embeddings.index])
    del texts_embeddings
    data.index.rename('id',inplace=True)
    print('data below: ')
    if args.debug:
        print(data)

    # Use embedding

    EMBEDDING_DIM = embeddings_index.get('a').shape[0]
    print('Embeddings dim: ' + str(EMBEDDING_DIM))
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    print("Num words: " + str(num_words))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    # Fill embedding matrix embedding_matrix[word_index] = embedding_vector
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        # Get the embedding for each word
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Not found are zeros
            embedding_matrix[i] = embedding_vector

    np.save(os.path.join(directory, 'embedding_matrix.npy'), embedding_matrix)

    '''
    print(embedding_matrix.shape)
    plt.plot(embedding_matrix[2])
    plt.plot(embedding_matrix[5])
    plt.plot(embedding_matrix[9])
    plt.title('Example vectors')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])plt.show()
    '''
    if args.debug:
        print(cpis)

    x_train_test, x_holdout, y_train_test, y_holdout = train_test_split(data,
            cpis, test_size=0.1)

    x_train_test.to_csv(os.path.join(directory, 'x_train_test_' + CPI_DATASET_FILE), header = True, sep='|')
    y_train_test.to_csv(os.path.join(directory, 'y_train_test_' + CPI_DATASET_FILE), header= True, sep='|')
    x_holdout.to_csv(os.path.join(directory, 'x_holdout_' + CPI_DATASET_FILE), header = True, sep='|')
    y_holdout.to_csv(os.path.join(directory, 'y_holdout_' + CPI_DATASET_FILE), header = True, sep='|')
    # I need to keep  number of words for later runs. 
    with open(os.path.join(directory, 'num_words.txt'), 'w') as f:
        f.write(str(num_words))
    print('Done all writing')
print('load data and embeddings from file/')
num_words = 0
with open(os.path.join(directory, 'num_words.txt'), 'r') as f:
    num_words = int(f.readline())
embedding_matrix = np.load(os.path.join(directory, 'embedding_matrix.npy'))

# read x_train back in from file:
column_names = ['title_a', 'title_b', 'texts_embeddings']
x_train_test_read = pd.read_csv(os.path.join(directory, 'x_train_test_' + CPI_DATASET_FILE), sep='|', index_col='id')

y_train_test_read = pd.read_csv(os.path.join(directory, 'y_train_test_' + CPI_DATASET_FILE), sep='|', index_col='id')

x_train, x_test,  y_train, y_test = train_test_split(x_train_test_read, y_train_test_read, test_size=0.2)
print(x_train_test_read.head())
'''
x_train_titles = x_train.loc[:,x_train.columns.isin(['title_a', 'title_b'])]
x_train_emebeddings = x_train.loc[:,~x_train.columns.isin(['title_a', 'title_b'])]
x_test_titles = x_test.loc[:,x_train.columns.isin(['title_a', 'title_b'])]
x_test_emebeddings = x_test.loc[:,~x_train.columns.isin(['title_a', 'title_b'])]

'''

GLOVE_EMBEDDING_DIMS = 100
model = Sequential()
model.add(Embedding(num_words, GLOVE_EMBEDDING_DIMS, input_length=MAX_SEQUENCE_LENGTH, weights= [embedding_matrix], trainable=False))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['mae', rmse])
print('head x_train:')
print(x_train.head())
filepath = os.path.join(directory, "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=False, mode='max')
history = model.fit(x_train.drop(['title_a', 'title_b'], axis=1), y_train, validation_split=0.1, epochs = 5)
model.save(os.path.join(directory, 'saved-model-'+CPI_DATASET))
# Save history to file.
with open(os.path.join(directory, 'history_' + CPI_DATASET + '.json'), 'w') as f:
    json.dump(history.history, f)
model_json = model.to_json()

# Serialize model to JSON
model_json = model.to_json()
with open(os.path.join(directory, 'model_' + 'DC1DMPLSTMD' + CPI_DATASET + '.json'), 'w') as json_model_file:
   json_model_file.write(model_json) 
pred = model.predict(x_test.drop(['title_a', 'title_b'], axis=1))
results = pd.concat([x_test['title_a'], x_test['title_b'] , y_test], axis=1, join_axes=[y_test.index])

results['pred_avg_cpi'] = pred
results.to_csv(os.path.join(directory, 'results_conv_glove_100d_' + CPI_DATASET_FILE), sep='|', index=False)

if(args.debug):
    print(results)
    pred = model.predict(x_train.drop(['title_a', 'title_b'], axis=1))
    train_results = pd.concat([x_train['title_a'], x_train['title_b'] , y_train], axis=1, join_axes=[y_train.index])

    train_results['pred_avg_cpi'] = pred
    train_results.to_csv(os.path.join(directory, 'train_results_conv_glove_100d_' + CPI_DATASET_FILE), sep='|', index=False)

loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
mae_history_val = history.history['val_mean_absolute_error']
mae_history = history.history['mean_absolute_error']

plt.plot(epochs, mae_history, 'bo', label='Training mae')
plt.plot(epochs, mae_history_val, 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.legend()
plt.show()


