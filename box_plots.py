import os
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt

CPI_BASE_DIR = '/home/pablo/FYP/'
CPI_RESULTS_FILENAME = 'final_results_conv_glove_100d_cpi_tokenized_paragraphs_1000_sample_count_5.csv'

# directory-> run title
model_runs = {'1000_with_titles_convlstm2x_redo_200_tokenized_results': 'Baseline 1DConvnet LSTM',
        '1000_with_titles_siamese_200_tokenized_results': 'Siamese LSTM',
        '1000_with_titles_siamese_2x_200_tokenized_results': 'Siamese 1DConvnet LSTM',
        '1000_with_titles_siamese_2x_no_dropout_200_tokenized_results': 'Siamese 1DConvnet LSTM No Dropout',
        '1000_with_titles_siamese_2x_50_tokenized_results': 'Siamese 1DConvnet LSTM Len 50'
        }

for directory, title in model_runs.items():
    full_dir = os.path.join(os.path.join(CPI_BASE_DIR, directory), CPI_RESULTS_FILENAME)
    # load results file
    results = pd.DataFrame()
    results = read_csv(full_dir, sep='|')
    # descriptive stats
    print(results.describe())
    # box and whisker plot
    ax = results.boxplot()
    plt.ylim([0, 1])
    plt.show() 
    # histogram
    results.hist()
    plt.show()

