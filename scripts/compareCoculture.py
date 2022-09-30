# %%
import pickle
import random
import pandas as pd
# from cellMorphHelper import procrustes
import cellMorphHelper
import cellMorph
import numpy as np
import datetime

import matplotlib.pyplot as plt
import umap

from skimage.io import imread

from sklearn.cluster import KMeans
# %%
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
esamPos = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))
coculture = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E7.pickle',"rb"))
# %% Constrain to low confluency
esamNeg = [cell for cell in esamNeg if cell.date < datetime.datetime(2022, 4, 8, 16, 0)]
esamPos = [cell for cell in esamPos if cell.date < datetime.datetime(2022, 4, 8, 16, 0)]
coculture = [cell for cell in coculture if cell.date < datetime.datetime(2022, 4, 8, 16, 0)]
# Filter color
esamNeg = [cell for cell in esamNeg if cell.color=='red']
esamPos = [cell for cell in esamPos if cell.color=='green']
coculture = [cell for cell in coculture if cell.color in ['green', 'red']]

# Filter borders
esamNeg = [cell for cell in esamNeg if cellMorphHelper.clearEdgeCells(cell) == 1]
esamPos = [cell for cell in esamPos if cellMorphHelper.clearEdgeCells(cell) == 1]
coculture = [cell for cell in coculture if cellMorphHelper.clearEdgeCells(cell) == 1]

# %%
# Subset some cells
random.seed(1234)
coculturePerm = random.sample(range(len(coculture)), len(coculture))
esamNegPerm = random.sample(range(len(esamNeg)), len(esamNeg))
esamPosPerm = random.sample(range(len(esamPos)), len(esamPos))
nDesired = 5000

cocultureSub = [coculture[x] for x in coculturePerm if coculture[x].color!='NaN']
esamNegSub = [esamNeg[x] for x in esamNegPerm if esamNeg[x].color=='red']
esamPosSub = [esamPos[x] for x in esamPosPerm if esamPos[x].color=='green']

cocultureSub = cocultureSub[0:nDesired]
esamNegSub = esamNegSub[0:nDesired]
esamPosSub = esamPosSub[0:nDesired]

origPerim = cocultureSub[0].perimAligned.copy()

# %% Align perimeters to each other
scalingBool = 0
referencePerim = esamNegSub[0].perimInt
c = 1


for cell in esamNegSub:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=scalingBool)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)


for cell in cocultureSub:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=scalingBool)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)

for cell in esamPosSub:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=scalingBool)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
   
# %% Build dataframe
labels = ['Coculture ESAM +' if x.color == 'red' else 'Coculture ESAM -' for x in cocultureSub]+ \
['Monoculture ESAM -' for x in range(len(esamNegSub))]+ \
['Monoculture ESAM +' for x in range(len(esamPosSub))]

label2Color = {'Monoculture ESAM -': 'red', 'Monoculture ESAM +': 'green', \
    'Coculture ESAM -': 'gold', 'Coculture ESAM +': 'purple'}
y = []
for label in labels:
    y.append(label2Color[label])

allCells = cocultureSub+esamNegSub+esamPosSub
X = []
for cell in allCells:
    X.append(cell.perimAligned.ravel())
X = np.array(X)
# X = pd.DataFrame(X)

# %% UMAP
fit = umap.UMAP()
u = fit.fit_transform(X)
# %%
fontSize = 20
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for label in np.unique(labels):
    if 'e' in label:
        labelIdx = np.where(np.array(labels)==label)
        ux = u[labelIdx,0]
        uy = u[labelIdx,1]
        ax.scatter(ux, uy, s=5, c=label2Color[label], alpha=0.5, label=label)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('ESAM Morphology')
ax.title.set_size(      fontSize)
ax.xaxis.label.set_size(fontSize)
ax.yaxis.label.set_size(fontSize)
ax.legend(markerscale=4)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_yticks([])
ax.set_xticks([])
fig.savefig('../results/figs/esamCoMonoUMAP.png', dpi=600)
# %% TSNE
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
pcax = pca.fit_transform(X)

# %%
fontSize = 20
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for label in np.unique(labels):
    labelIdx = np.where(np.array(labels)==label)
    ux = pcax[labelIdx,0]
    uy = pcax[labelIdx,1]
    ax.scatter(ux, uy, s=5, c=label2Color[label], alpha=0.5, label=label)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_title('ESAM Morphology')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.title.set_size(      fontSize)
ax.xaxis.label.set_size(fontSize)
ax.yaxis.label.set_size(fontSize)
ax.legend(markerscale=4)
ax.set_yticks([])
ax.set_xticks([])
fig.savefig('../results/figs/esamCoMonoPCA.png', dpi=600)
# %%
cell = allCells[251]
mask = cell.mask
a = np.where(mask==True)
bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
im = imread(cell.composite)
im = im[bbox[0]:bbox[1], bbox[2]:bbox[3]]

fig, ax = plt.subplots(2,2)
fig.set_size_inches(6, 6)
ax[0,0].imshow(im, origin='lower')
ax[0,0].xaxis.set_ticklabels([])
ax[0,0].yaxis.set_ticklabels([])
ax[0,0].set_yticks([])
ax[0,0].set_xticks([])
ax[0,0].set_title('Identify Cell')
ax[0,0].set_box_aspect(1)

ax[0,1].plot(cell.perimeter[:,0], cell.perimeter[:,1])
ax[0,1].xaxis.set_ticklabels([])
ax[0,1].yaxis.set_ticklabels([])
ax[0,1].set_yticks([])
ax[0,1].set_xticks([])
ax[0,1].set_title('Find Perimeter')
ax[0,1].set_box_aspect(1)

pI = cell.perimInt.copy()
pI = pI - np.mean(pI, axis = 0)
ax[1,0].scatter(pI[:,0], pI[:,1], s = 0.5)
ax[1,0].set_xlim([-18,18])
ax[1,0].set_ylim([-18,18])
ax[1,0].xaxis.set_ticklabels([])
ax[1,0].yaxis.set_ticklabels([])
ax[1,0].set_yticks([])
ax[1,0].set_xticks([])
ax[1,0].set_title('Interpolate Perimeter')
ax[1,0].set_box_aspect(1)

ax[1,1].scatter(cell.perimAligned[:,0], cell.perimAligned[:,1], s = 0.5)
ax[1,1].set_xlim([-20,20])
ax[1,1].set_ylim([-18,18])
ax[1,1].xaxis.set_ticklabels([])
ax[1,1].yaxis.set_ticklabels([])
ax[1,1].set_yticks([])
ax[1,1].set_xticks([])
ax[1,1].set_title('Align and Center')
ax[1,1].set_box_aspect(1)

fig.savefig('../results/figs/perimeterFeatureMethod.png', dpi=600)
# %%
esamNegIdx = np.where(np.array(labels) == 'Monoculture ESAM -')[0]
esamPosIdx = np.where(np.array(labels) == 'Monoculture ESAM +')[0]
esamCoPosIdx = np.where(np.array(labels) == 'Coculture ESAM +')[0]
esamCoNegIdx = np.where(np.array(labels) == 'Coculture ESAM -')[0]


XNeg = X[esamNegIdx, :]
XPos = X[esamPosIdx, :]
XCoPos = X[esamCoPosIdx, :]
XCoNeg = X[esamCoNegIdx, :]

avgPos = np.mean(XPos, axis=0)
avgNeg = np.mean(XNeg, axis=0)
avgCoPos = np.mean(XCoPos, axis=0)
avgCoNeg = np.mean(XCoNeg, axis=0)

fig, ax = plt.subplots(2,2)
fig.set_size_inches(12, 6)
ax[0,0].scatter(avgPos[::2], avgPos[1::2], s=5 , c = 'green')
ax[0,1].scatter(avgNeg[::2], avgNeg[1::2], s=5 , c = 'red')
ax[1,0].scatter(avgCoPos[::2], avgCoPos[1::2], s=5 , c = 'green')
ax[1,1].scatter(avgCoNeg[::2], avgCoNeg[1::2], s=5 , c = 'red')

# %% Quick and dirty logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

monoIdx = [1 if 'mono' in label.lower() else 0 for label in labels]
monoIdx = np.array(monoIdx) == 1
XMono = X[monoIdx,:]
yMono = np.array(labels)[monoIdx]

df = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(XMono, yMono, test_size=0.33, random_state=1234)
clf = LogisticRegression(solver="liblinear", random_state=1234, C=1e-6,max_iter=1e7).fit(X_train, y_train)
roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])