# %% [markdown]
"""
This is a notebook to compare high and low eccentric cells to see if there are notable differences via UMAP representation
"""
# %%
import sys
sys.path.append('../scripts')
import pickle
import cellMorphHelper
import cellMorph
import datetime
import matplotlib.pyplot as plt
import numpy as np
import umap
import pandas as pd
import seaborn as sns
import random

from skimage.measure import label, regionprops, regionprops_table
# %%
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
esamPos = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))

esamNeg = cellMorphHelper.filterCells(esamNeg, confluencyDate=datetime.datetime(2022, 4, 8, 16, 0), color='red', edge=True)
esamPos = cellMorphHelper.filterCells(esamPos, confluencyDate=datetime.datetime(2022, 4, 8, 16, 0), color='green', edge=True)

esamNegOrig = esamNeg.copy()
esamPosOrig = esamPos.copy()
# %%
cells = esamNeg+esamPos
lowEcc, highEcc = [], []
lowEccThresh, highEccThresh = 0.4, 0.9
for cell in cells:
    region = regionprops(cell.mask.astype(np.uint8))
    if len(region)>1:
        region = sorted(region, key = lambda allprops: allprops.area)
    region = region[0]
    if region.eccentricity <= lowEccThresh:
        lowEcc.append(cell)
    elif region.eccentricity >= highEccThresh:
        highEcc.append(cell)
# %%
scalingBool = 0
referencePerim = highEcc[0].perimInt
c = 1

for cell in lowEcc:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=scalingBool)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)

for cell in highEcc:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=scalingBool)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
# %%
random.seed(1234)
random.shuffle(lowEcc)
cells = lowEcc+highEcc[0:len(lowEcc)]
nCells = len(lowEcc)
labels = ['low' for i in range(nCells)]+['high' for i in range(nCells)]

X = []
for cell in cells:
    X.append(cell.perimAligned.ravel())
X = np.array(X)
# %%
fit = umap.UMAP()
u = fit.fit_transform(X)
# %%
fontSize = 20
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)

label2Color = {'low': 'red', 'high': 'blue'}
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for label in np.unique(labels):
    labelIdx = np.where(np.array(labels)==label)
    ux = u[labelIdx,0]
    uy = u[labelIdx,1]
    ax.scatter(ux, uy, s=5, c=label2Color[label], alpha=0.5, label=label)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title(f'ESAM Perimeter Morphology Eccentricity > {eccNum:0.2f}')
ax.title.set_size(      fontSize)
ax.xaxis.label.set_size(fontSize)
ax.yaxis.label.set_size(fontSize)
ax.legend(markerscale=4)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_yticks([])
ax.set_xticks([])