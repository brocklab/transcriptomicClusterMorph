# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline
from scipy import interpolate
from scipy.signal import savgol_filter
# %%
modelRes = pd.read_csv('../../data/esamMonoSegmented/logs/modelCellUncrop-1664832998.log', header=None)
modelRes.columns = ['model', 'time', 'trainAcc','trainLoss', 'valAcc', 'valLoss']

for metric in ['trainAcc','trainLoss', 'valAcc', 'valLoss']:
    modelRes[metric] = savgol_filter(modelRes[metric], 17, 3)

iters = list(range(1, modelRes.shape[0]*10, 10))
modelRes.head()
# %%
plt.figure(figsize=(16,11))
matplotlib.rcParams.update({'font.size': 17})

plt.subplot(2,2,1)
plt.plot(iters, modelRes['trainLoss'])
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.grid()

plt.subplot(2,2,2)
plt.plot(iters, modelRes['trainAcc'])
plt.xlabel('Iteration')
plt.ylabel('Training Accuracy')
plt.grid()

plt.subplot(2,2,3)
plt.plot(iters, modelRes['valLoss'])
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.grid()

plt.subplot(2,2,4)
plt.plot(iters, modelRes['valAcc'])
plt.xlabel('Iteration')
plt.ylabel('Validation Accuracy')
plt.grid()

plt.suptitle('Small CNN Training Results')
# plt.savefig('../../results/figs/smallCNN.png', dpi=600)
plt.show()


# %%
