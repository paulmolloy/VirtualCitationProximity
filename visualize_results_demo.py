import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

models = ('Baseline 1DConvnet LSTM', 'Siamese LSTM', 'Siamese Convnet+LSTM', 'Siamese Convnet+LSTM No Dropout', 'Siamese Convnet+LSTM Len 50')
y_pos = np.arange(len(models))

eval_maes = [0.005941064122001566, 0.005516218107877257, 0.005732106241843341, 0.005518515549014526, 0.005458833505198094]
fig, ax = plt.subplots()
plt.bar(y_pos, eval_maes, align='center', alpha=0.5)
plt.xticks(y_pos, models)
ax.set_title('MAE Results After 5 Epochs trained on 176841 CPI Pairs for 1k Articles')
plt.xlabel('Model')
plt.ylabel('Mean Average Error')



plt.show()



