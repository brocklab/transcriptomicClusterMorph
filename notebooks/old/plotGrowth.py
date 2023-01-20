# %%
import sys
sys.path.append('../scripts')
import cellMorphHelper
from cellMorph import imgSegment

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from skimage.measure import regionprops
from skimage.io import imread

import datetime
# %%
classes = pickle.load(open('../data/fullClassificationAccuracy.pickle',"rb"))
# %%
well = 'E7'

classes = [classification for classification in classes if classification.well == 'E7']

# %%
actualGrowth = {}
predictedGrowth = {}
for classification in classes:
    date = classification.date

    pred0 = sum(np.array(classification.predClasses) == 0)
    pred1 = sum(np.array(classification.predClasses) == 1)
    
    actual0 = sum(np.array(classification.actualClasses) == 0)
    actual1 = sum(np.array(classification.actualClasses) == 1)

    if date not in predictedGrowth.keys():
        predictedGrowth[date] = [0, 0]
        actualGrowth[date] = [0, 0]
    
    predictedGrowth[date][0] += pred0
    predictedGrowth[date][1] += pred1

    actualGrowth[date][0] += actual0
    actualGrowth[date][1] += actual1


predictedGrowth = pd.DataFrame(predictedGrowth).T.reset_index()
predictedGrowth.columns = ['date', 0, 1]

predictedGrowth['date'] = predictedGrowth['date'] - min(predictedGrowth['date'])
predictedGrowth['date'] = predictedGrowth['date'].dt.total_seconds()/3600

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize = (8, 6))
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.scatter(predictedGrowth['date'], predictedGrowth[0], c = 'red', label = 'ESAM (-)')
plt.scatter(predictedGrowth['date'], predictedGrowth[1], c = 'green', label = 'ESAM (+)')
plt.xlabel('Hours')
plt.ylabel('Predicted Growth')
plt.savefig('../results/figs/predictedCocultureGrowth.png', dpi=600)
plt.show()

actualGrowth = pd.DataFrame(actualGrowth).T.reset_index()
actualGrowth.columns = ['date', 0, 1]

plt.figure()
plt.scatter(actualGrowth['date'], actualGrowth[0], c = 'red')
plt.scatter(actualGrowth['date'], actualGrowth[1], c = 'green')
plt.show()
# %%
dates = {}
for c in classes:
    imNum = c.pcImg.split('_')[2]
    date = c.date
    if date not in dates.keys():
        dates[date] = []
    dates[date].append(imNum)

nIms = []
for date in dates.keys():
    nIms.append(len(set(dates[date])))