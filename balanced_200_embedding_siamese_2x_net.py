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

# balanced_200_embedding_siamese_2x_net trains the siamese 1DConvnet + LSTM model on the larger balanced data-set.
# Saves history after every epoch due to the long running time.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', dest='debug', help= 'For debugging script use test file.', action='store_true')
parser.add_argument('-e', '--reload_embeddings', dest='reload_embeddings', help= 'Reload embeddings from file.', action='store_true')

parser.set_defaults(debug=False)
args = parser.parse_args()
MAX_NUM_WORDS= 50000
MAX_SEQUENCE_LENGTH = 200 # each title could be 100 tokens each.
GLOVE_PATH = '/home/pablo/FYP/glove.6B/glove.6B.100d.txt'
CPI_DATASET_DIR = '/home/pablo/FYP/balanced_50000_with_titles_siamese_2x_200_tokenized_results/'
CPI_DATASET = 'cpi_tokenized_balanced_paragraphs_50000_sample'
CPI_DATASET_FILE = CPI_DATASET + '.csv' 
CPI_DATASET_PATH = os.path.join(CPI_DATASET_DIR, CPI_DATASET_FILE)

TEST_DIR = '/home/pablo/FYP/test'

TINY_TEST_FILE = 'cpi_tokenized_200_test.csv'
NO_AVG_TEST_FILE = 'test_cpi_tokenized_200_1000_sample.csv'
TEST_FILE = 'cpi_tokenized_paragraphs_1000_sample_count_5.csv'
TEST_DATASET_PATH = os.path.join(TEST_DIR, TEST_FILE)

def rmse (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred -y_true), axis=-1))

column_names = ['id', 'title_a', 'title_b', 'dist', 'count', 'id_a', 'id_b', 'cpi', 'avg_cpi', 'cpi_count_pt4', 'tokenized_title_a', 'tokenized_text_a', 'tokenized_title_b', 'tokenized_text_b']
print('Debug = ' + str(args.debug))
if(args.debug):
    directory = TEST_DIR
    dataset_path = TEST_DATASET_PATH
else:
    directory = CPI_DATASET_DIR
    dataset_path = CPI_DATASET_PATH

df_sample = pd.read_csv(dataset_path,sep='|', nrows=10, names=column_names)
print(df_sample.head())
df_sample_size = df_sample.memory_usage(index=False).sum();

# chunk size of 1 gb of csv text.
Gb = 1000000000
chunk_size = (Gb/ df_sample_size)/10
chunk_size = int(chunk_size//1)
print(chunk_size)
if args.reload_embeddings:
    #data = pd.read_csv(dataset_path, sep='|',  names = column_names)
    data_iter = pd.read_csv(dataset_path, iterator=True, chunksize=chunk_size, sep='|',  names = column_names)
    # Format example: 
    #
    # 2|United States|Lily van den Broecke雲|13275|1|1919370|36828443|0.000298404574868282|United States|This is the article about usa|Lily van den Broecke雲| This person sounds雲 dutch yū.


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
    for data in data_iter:
        cpi_problems = data[data['cpi'] == np.nan]
        if len(cpi_problems) != 0:
                print('Some cpis are Nan clean data some more.')
        print('title_a:')
        print(data['title_a'].head())
        data = data.replace(np.nan, '', regex=True)
        # tokenized_titles_ab = data['tokenized_title_a'].map(str) + ' ' + data['tokenized_title_b']
        tokenized_texts_ab = data['tokenized_text_a'].map(str) + ' ' + data['tokenized_text_b']
        print(len(data))
        print(tokenized_texts_ab.head())


        # Creates internal word index based on frequency 
        tokenizer.fit_on_texts(tokenized_texts_ab)
        del tokenized_texts_ab
        print('Finish fit on texts')

    # Get word index
    word_index = tokenizer.word_index
    print('Len word_idex:' + str(len(word_index)))
    if args.debug:
        print('Word index: ' + str(word_index))
    is_first = True

    data_iter = pd.read_csv(dataset_path, iterator=True, chunksize=chunk_size, sep='|',  names = column_names)
    # Format example: 
    for data in data_iter:
        print('embedding current data')
        print(data.head())
        # Transforms texts into sequence of integers based on the tokenizers 
        # word index.
        sequences_texts_a = tokenizer.texts_to_sequences(data['tokenized_text_a'].map(str))
        sequences_texts_b = tokenizer.texts_to_sequences(data['tokenized_text_b'].map(str))

        # add zeros to the vector to make it len 200
        p_a = pad_sequences(tokenizer.texts_to_sequences(data['tokenized_text_a'].map(str)), 
                maxlen=MAX_SEQUENCE_LENGTH)
        p_b = pad_sequences(tokenizer.texts_to_sequences(data['tokenized_text_b'].map(str)), 
                maxlen=MAX_SEQUENCE_LENGTH)
        texts_embeddings = pd.concat([pd.DataFrame(p_a), pd.DataFrame(p_b)], axis=1)
        texts_embeddings.columns = [x for x in range(400)]
        
        del p_a
        del p_b

        print('made embeddings frame')
        cpis = data['avg_cpi']
        cpis.index.rename('id',inplace=True)
        print('Got cpis etc')
        print('titlea titleb')
        print(data['title_a'].head())
        print(data['title_b'].head())
        data = pd.concat( [data['title_a'], data['title_b'], texts_embeddings.set_index(data.index.copy())], axis=1)
        del texts_embeddings
        data.index.rename('id',inplace=True)
        print('data below: ')
        if args.debug:
            print(data.head())

        x_train_test, x_holdout, y_train_test, y_holdout = train_test_split(data,
                cpis, test_size=0.1)
        if is_first:
            x_train_test.to_csv(os.path.join(directory, 'x_train_test_' + CPI_DATASET_FILE), header = is_first, sep='|')
            y_train_test.to_csv(os.path.join(directory, 'y_train_test_' + CPI_DATASET_FILE), header= is_first, sep='|')
            x_holdout.to_csv(os.path.join(directory, 'x_holdout_' + CPI_DATASET_FILE), header = is_first, sep='|')
            y_holdout.to_csv(os.path.join(directory, 'y_holdout_' + CPI_DATASET_FILE), header = is_first, sep='|')
            is_first = False   
        else:
            x_train_test.to_csv(os.path.join(directory, 'x_train_test_' + CPI_DATASET_FILE), mode='a',  header = is_first, sep='|')
            y_train_test.to_csv(os.path.join(directory, 'y_train_test_' + CPI_DATASET_FILE), mode='a', header= is_first, sep='|')
            x_holdout.to_csv(os.path.join(directory, 'x_holdout_' + CPI_DATASET_FILE), mode='a', header = is_first, sep='|')
            y_holdout.to_csv(os.path.join(directory, 'y_holdout_' + CPI_DATASET_FILE), mode='a', header = is_first, sep='|')

        print(x_train_test.head())
        print('Written')


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

    # I need to keep  number of words for later runs. 
    with open(os.path.join(directory, 'num_words.txt'), 'w') as f:
        f.write(str(num_words))
    print('Done all writing')


# Columns titled "0" "199" are the emdeddings for  text_a
a_columns = [str(x) for x in range(200)]
# Columns titled "200" to "399" are the embeddings for text_b
b_columns = [str(x) for x in range(200, 400)]

print('load data and embeddings from file/')
num_words = 0
with open(os.path.join(directory, 'num_words.txt'), 'r') as f:
    num_words = int(f.readline())
embedding_matrix = np.load(os.path.join(directory, 'embedding_matrix.npy'))


GLOVE_EMBEDDING_DIMS = 100

# Setup model.
a_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
b_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

shared_model = Sequential()
shared_model.add(Embedding(num_words, GLOVE_EMBEDDING_DIMS, input_length=MAX_SEQUENCE_LENGTH, weights= [embedding_matrix], trainable=False))
shared_model.add(Dropout(0.2))
shared_model.add(Conv1D(64, 5, activation='relu'))
shared_model.add(MaxPooling1D(pool_size=4))
shared_model.add(LSTM(100))

# Overarching model.
a_features = shared_model(a_input)
b_features = shared_model(b_input)

siamese_model = concatenate([a_features, b_features])
siamese_model = Dropout(0.2)(siamese_model)
preds = Dense(1,  activation='sigmoid')(siamese_model)

model = Model(inputs=[a_input, b_input], outputs=preds)
model.compile(optimizer='adam', loss='mse', metrics=['mae', rmse])

filepath = os.path.join(directory, "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=False, mode='max')
num_epochs = 5
for epoch in range(num_epochs):

    # read x_train back in from file:
    column_names = ['title_a', 'title_b', 'texts_embeddings']
    x_train_test_iter = pd.read_csv(os.path.join(directory, 'x_train_test_' + CPI_DATASET_FILE), sep='|', index_col='id', iterator=True, chunksize=chunk_size)

    y_train_test_iter = pd.read_csv(os.path.join(directory, 'y_train_test_' + CPI_DATASET_FILE), sep='|', index_col='id', iterator=True, chunksize=chunk_size)

    for X, y in zip(x_train_test_iter, y_train_test_iter):


        print("Training on next chunk")
        print(X.head())
        print(X[b_columns].head())
        print(X[a_columns].head())
        # Have to roll my own epochs in the loop.
        history = model.fit([X[a_columns], X[b_columns]], y, validation_split=0.1, epochs = 1)
    # Save history to file.
    with open(os.path.join(directory, str(epoch) + '_history_' + CPI_DATASET + '.json'), 'w') as f:
        json.dump(history.history, f)


model.save(os.path.join(directory, 'saved-model-'+CPI_DATASET))
# Save history to file.
with open(os.path.join(directory, 'history_' + CPI_DATASET + '.json'), 'w') as f:
    json.dump(history.history, f)
model_json = model.to_json()

# Serialize model to JSON
model_json = model.to_json()
with open(os.path.join(directory, 'model_' + 'siamese_2x' + CPI_DATASET + '.json'), 'w') as json_model_file:
   json_model_file.write(model_json) 
pred = model.predict([x_test[a_columns], x_test[b_columns]])
results = pd.concat([x_test['title_a'], x_test['title_b'] , y_test], axis=1, join_axes=[y_test.index])
results['pred_avg_cpi'] = pred
results.to_csv(os.path.join(directory, 'results_conv_glove_100d_' + CPI_DATASET_FILE), sep='|', index=False)

if(args.debug):
    print(results)

    pred = model.predict([x_train[a_columns], x_train[b_columns]])
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


