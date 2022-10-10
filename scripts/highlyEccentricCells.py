# %% [markdown]
"""
This is an analysis of perimeters of only highly eccentric cells, based on the hypothesis
that these cells are the ones driving differences. 
"""
# %%
import pickle
import cellMorphHelper
import cellMorph
import datetime
import matplotlib.pyplot as plt
import numpy as np
import umap
import pandas as pd
import seaborn as sns

from skimage.measure import label, regionprops, regionprops_table
# %%
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
esamPos = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))

esamNeg = cellMorphHelper.filterCells(esamNeg, confluencyDate=datetime.datetime(2022, 4, 8, 16, 0), color='red', edge=True)
esamPos = cellMorphHelper.filterCells(esamPos, confluencyDate=datetime.datetime(2022, 4, 8, 16, 0), color='green', edge=True)

esamNegOrig = esamNeg.copy()
esamPosOrig = esamPos.copy()
# %%
eccPos, eccNeg = [], []
for cell in esamNeg:
    region = regionprops(cell.mask.astype(np.uint8))
    if len(region)>1:
        region = sorted(region, key = lambda allprops: allprops.area)
    region = region[0]
    eccNeg.append(region.eccentricity)

for cell in esamPos:
    region = regionprops(cell.mask.astype(np.uint8))
    if len(region)>1:
        region = sorted(region, key = lambda allprops: allprops.area)
    region = region[0]
    eccPos.append(region.eccentricity)

plt.subplot(121)
plt.hist(eccNeg)
plt.title('Eccentricity of ESAM (-)')
plt.subplot(122)
plt.hist(eccPos)
plt.title('Eccentricity of ESAM (+)')

# %% Filter eccentricity
eccNum = 0.9

esamNeg = [cell for cell, ecc in zip(esamNegOrig, eccNeg) if ecc>=eccNum]
esamPos = [cell for cell, ecc in zip(esamPosOrig, eccNeg) if ecc>=eccNum]

# %%
properties=['area', 'perimeter', 'feret_diameter_max', 'solidity']
esamNegProperties, esamPosProperties = {property: [] for property in properties}, {property: [] for property in properties}

for cell in esamNeg:
    computedProps = regionprops_table(cell.mask.astype(np.uint8), properties = properties)
    for property in properties:
        esamNegProperties[property].append(computedProps[property][0])    

for cell in esamPos:
    computedProps = regionprops_table(cell.mask.astype(np.uint8), properties = properties)
    for property in properties:
        esamPosProperties[property].append(computedProps[property][0])    

# Concatenate
esamNegProperties = pd.DataFrame(esamNegProperties)
esamPosProperties = pd.DataFrame(esamPosProperties)

esamNegProperties['phenotype'] = 'esamNeg'
esamPosProperties['phenotype'] = 'esamPos'

cellProperties = pd.concat([esamPosProperties, esamNegProperties]).reset_index().drop(['index'], axis=1)
# %%
sns.set(font_scale=1.25)
hueDict = {'esamPos': 'green', 'esamNeg': 'red'}
pp = sns.pairplot(cellProperties, hue="phenotype", palette=hueDict)
fig = pp.fig
fig.savefig(f'../results/figs/morphPairEsam_{eccNum:0.2f}.png')
# %%
scalingBool = 0
referencePerim = esamNeg[0].perimInt
c = 1

for cell in esamNeg:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=scalingBool)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)

for cell in esamPos:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=scalingBool)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)


# %% Build dataframe
labels = ['Monoculture ESAM -' for x in range(len(esamNeg))]+ \
['Monoculture ESAM +' for x in range(len(esamPos))]

label2Color = {'Monoculture ESAM -': 'red', 'Monoculture ESAM +': 'green', \
    'Coculture ESAM -': 'gold', 'Coculture ESAM +': 'purple'}
y = []
for label in labels:
    y.append(label2Color[label])

allCells = esamNeg+esamPos
X = []
for cell in allCells:
    X.append(cell.perimAligned.ravel())
X = np.array(X)
# %%
fit = umap.UMAP()
u = fit.fit_transform(X)
# %%
sns.reset_orig()
fontSize = 20
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)


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
# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
# kmeans.labels_

for label in kmeans.labels_.unique():
    labelIdx = np.where(kmeans.labels_==label)
    plt.figure()
    plt.scatter(u[labelIdx,0], u[labelIdx,1])
    plt.title(label)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
clf = LogisticRegression(solver="liblinear", random_state=1234, C=1e-6,max_iter=1e7).fit(X_train, y_train)
roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
# %%
