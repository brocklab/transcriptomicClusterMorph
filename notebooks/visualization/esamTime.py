# %%
import pickle
import random
import pandas as pd
# from cellMorphHelper import procrustes
import src.data.cellMorphHelper as cellMorphHelper
import src.data.cellMorph as cellMorph
import numpy as np
import datetime

import matplotlib.pyplot as plt
import umap

from skimage.io import imread

from sklearn.cluster import KMeans
# %%
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
esamPos = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))
# %% Constrain to low confluency
esamNeg = [cell for cell in esamNeg if cell.date < datetime.datetime(2022, 4, 8, 16, 0)]
esamPos = [cell for cell in esamPos if cell.date < datetime.datetime(2022, 4, 8, 16, 0)]
# Filter color
esamNeg = [cell for cell in esamNeg if cell.color=='red']
esamPos = [cell for cell in esamPos if cell.color=='green']

# Filter borders
esamNeg = [cell for cell in esamNeg if cellMorphHelper.clearEdgeCells(cell) == 1]
esamPos = [cell for cell in esamPos if cellMorphHelper.clearEdgeCells(cell) == 1]

# %% Subset some cells
random.seed(1234)
esamNegPerm = random.sample(range(len(esamNeg)), len(esamNeg))
esamPosPerm = random.sample(range(len(esamPos)), len(esamPos))
nDesired = 5000

esamNegSub = [esamNeg[x] for x in esamNegPerm if esamNeg[x].color=='red']
esamPosSub = [esamPos[x] for x in esamPosPerm if esamPos[x].color=='green']

esamNegSub = esamNegSub[0:nDesired]
esamPosSub = esamPosSub[0:nDesired]

# %% Align perimeters to each other
scalingBool = 0
referencePerim = esamNegSub[0].perimInt
c = 1

for cell in esamNegSub:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=scalingBool)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)

for cell in esamPosSub:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=scalingBool)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
# %%
labels = ['Monoculture ESAM -' for x in range(len(esamNegSub))]+ \
['Monoculture ESAM +' for x in range(len(esamPosSub))]

label2Color = {'Monoculture ESAM -': 'red', 'Monoculture ESAM +': 'green', \
    'Coculture ESAM -': 'gold', 'Coculture ESAM +': 'purple'}
y = []
for label in labels:
    y.append(label2Color[label])

allCells = esamNegSub+esamPosSub
X = []
for cell in allCells:
    X.append(cell.perimAligned.ravel())
X = np.array(X)
# X = pd.DataFrame(X)
# %% Labels/dates for only esam - or pos
X = []
for cell in esamPosSub:
    X.append(cell.perimAligned.ravel())
for cell in esamNegSub:
    X.append(cell.perimAligned.ravel())
X = np.array(X)
dates = [cell.date for cell in esamNegSub+esamPosSub]
# %% UMAP
fit = umap.UMAP()
u = fit.fit_transform(X)
# %%
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

matplotDates = mdates.date2num(dates)
fontSize = 20
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.scatter(u[0:5000,0], u[0:5000,1], s=5, c=matplotDates[0:5000], alpha=0.5, cmap='Greens', label='ESAM +')
ax.scatter(u[5000:,0],  u[5000:,1],  s=5, c=matplotDates[5000:len(matplotDates)], alpha=0.5,  cmap='Reds',   label='ESAM -')

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('ESAM Morphology')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.title.set_size(      fontSize)
ax.xaxis.label.set_size(fontSize)
ax.yaxis.label.set_size(fontSize)
ax.set_yticks([])
ax.set_xticks([])

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Monoculture ESAM +',
                          markerfacecolor='g', markersize=3),
                    Line2D([0], [0], marker='o', color='w', label='Monoculture ESAM -',
                    markerfacecolor='r', markersize=3)]

ax.legend(handles=legend_elements, markerscale=4)
fig.savefig('../results/figs/esamMonoTimeUMAP.png', dpi=600)

# %%
smap = ax.scatter(df['v'],df['u'],s=500,c=df.index,
                  edgecolors='none', marker='o', cmap=cmap)