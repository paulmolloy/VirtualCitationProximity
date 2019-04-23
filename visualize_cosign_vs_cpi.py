import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy import spatial
CPI_PATH = '/home/pablo/FYP/1000_with_titles_siamese_200_tokenized_results/'
EMBEDDINGS_PATH =  os.path.join(CPI_PATH + 'x_train_test_cpi_tokenized_paragraphs_1000_sample_count_5.csv') 
CPI_PAIRS_PATH = os.path.join(CPI_PATH + 'cpi_tokenized_paragraphs_1000_sample_count_5.csv')
CPI_VALUES_PATH = os.path.join(CPI_PATH + 'y_train_test_cpi_tokenized_paragraphs_1000_sample_count_5.csv')
column_names = ['id', 'title_a', 'title_b', 'dist', 'count', 'id_a', 'id_b', 'cpi', 'avg_cpi', 'cpi_count_pt4', 'tokenized_title_a', 'tokenized_text_a', 'tokenized_title_b', 'tokenized_text_b']

# Columns titled "0" "199" are the emdeddings for  text_a
a_columns = [str(x) for x in range(200)]
# Columns titled "200" to "399" are the embeddings for text_b
b_columns = [str(x) for x in range(200, 400)]

embedding_matrix = np.load(os.path.join(CPI_PATH, 'embedding_matrix.npy'))
y_train_test_read = pd.read_csv(CPI_VALUES_PATH, sep='|', index_col='id').head(10000)
x_train_test_read = pd.read_csv(EMBEDDINGS_PATH, sep='|', index_col='id').head(10000)
print(x_train_test_read)
print(x_train_test_read[a_columns].values)
print(x_train_test_read[b_columns].values)

print(x_train_test_read[a_columns].values - x_train_test_read[b_columns].values)
x_a = x_train_test_read[a_columns].values
x_b = x_train_test_read[b_columns].values
x_cosine = []
x_cosines_max = []
for i in range(len(x_a)):
    #print(x_a[i])
    similarities_sum = 0
    max_sum = 0
    count = 0 
    max_simiarities = []
    print("Pair:")

    MAX_COSINE_LENGTH = 25
    for j in range(len(x_a[i])-MAX_COSINE_LENGTH-1,len(x_a[i])):
        cur_max = -1

        #print(x_a[i])
        #print("embedding:")
        # Only compute cosine similarity for first 25 tokens.
        print('b')
        print(x_b[i][j:])
        if(x_a[i][j] != 0 and x_b[i][j] != 0 and j!= MAX_COSINE_LENGTH):
            # sum the cosine differences of all of the word vectors where the vec position is not 0 for both.   
            similarities_sum +=  1 - spatial.distance.cosine(embedding_matrix[x_a[i][j]], embedding_matrix[x_b[i][j]])
            count += 1;

            # Get the highest cosine score for each vector a from b
            #print('word:')
            for word_vec in x_b[i][-50:]:
                '''print('two embeddings: ')
                #print(embedding_matrix[x_a[i][j]])
                print(x_b[i])
                print(word_vec)
                print(embedding_matrix[word_vec])
                
               ''' 
                cur_sim =  1 - spatial.distance.cosine(embedding_matrix[x_a[i][j]], embedding_matrix[word_vec])

                if cur_sim > cur_max:
                    cur_max = cur_sim
                    print(cur_sim)
            max_sum += cur_max



    # average cosine diff of all words in the text 
    if count == 0:
        x_cosine.append(0)
        x_cosines_max.append(0)

    else:
        x_cosine.append(similarities_sum/count) 
        x_cosines_max.append(max_sum/count) 


y_train_test_read['cosine'] = x_cosine
y_train_test_read['cosine_max'] = x_cosines_max
fig, ax = plt.subplots()
print(y_train_test_read.head())
just_cpis = y_train_test_read['avg_cpi'].values
ax.scatter(y_train_test_read['cosine'], y_train_test_read['avg_cpi'])

ax.set_title('Ordered Comparison Cosine Similarity of GloVe Embeddings vs Avg CPI values from 1000 Articles Sample')
plt.ylim([0, 1])
plt.xlabel('Cosine Similarity')
plt.ylabel('Avg CPI')
plt.show()

fig, ax = plt.subplots()
print(y_train_test_read.head())
just_cpis = y_train_test_read['avg_cpi'].values
ax.scatter(y_train_test_read['cosine_max'], y_train_test_read['avg_cpi'])

ax.set_title('Max Cosine Similarity of GloVe Embeddings vs Avg CPI values from 1000 Articles Sample')
plt.xlabel('Cosine Similarity')
plt.ylabel('Avg CPI')
plt.show()

