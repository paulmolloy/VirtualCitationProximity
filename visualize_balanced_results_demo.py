import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

mae_history_val = [0.01328989020217826, 0.012876023521477526, 0.012673106597680034]
mae_history = [0.013168208323644869, 0.01309057256051911, 0.01314296151937038]
epochs = range(1, len(mae_history) + 1) 

plt.plot(epochs, mae_history, 'bo', label='Training mae')
plt.plot(epochs, mae_history_val, 'b', label='Validation mae')
plt.title('Training and Validation MAE Rebalanced Data-set')
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.legend()
plt.show()

