# %%
import pickle
import umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cellMorphHelper
# %%
featureRes = pickle.load(open('../results/allFeaturesTJ2201.pickle',"rb"))
colors = pickle.load(open('../results/TJ2201allColors.pickle',"rb"))
# %% Correct perimeters
perimeters = [featureRes[2][cell]['perimInt'] for cell in range(len(featureRes[2]))]

scalingBool = 0
referencePerim = perimeters[0]
c = 1

alignedPerimeters = []
for perimeter in perimeters:
    
    refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, perimeter, scaling=scalingBool)

    alignedPerimeters.append(currentPerim2 - np.mean(currentPerim2, axis=0))
# %%
features = pd.DataFrame(featureRes[0])
features = features.dropna(axis='columns').values

# Add perimeters
featurePerims = []
for cellNum in range(len(features)):
    featurePerims.append(features[cellNum].tolist() + alignedPerimeters[cellNum].ravel().tolist())

features = pd.DataFrame(featurePerims)

df = features.copy()
normalized_df=(df-df.min())/(df.max()-df.min())
normalized_df = normalized_df.dropna(axis='columns')
# %%
fit = umap.UMAP()
u = fit.fit_transform(normalized_df)
# %%
wells = [cell['well'] for cell in featureRes[2]]
colorWell = [f'{well}_{color}' for well, color in zip(wells, colors)]

uniqueColors = ['red', 'blue', 'green', 'magenta']
uniqueWells = set(wells)
colorDict = {well:color for well, color in zip(uniqueWells, uniqueColors)}
wellColor = [colorDict[well] for well in wells]


# %%
labels = []
for cell in colorWell:
    if cell == 'D2_green':
        labels.append('Monoculture ESAM +')
    elif cell == 'E2_red':
        labels.append('Monoculture ESAM -')
    elif cell == 'E7_green':
        labels.append('Coculture ESAM +')
    else:
        labels.append('Coculture ESAM -')
label2Color = {'Monoculture ESAM -': 'red', 'Monoculture ESAM +': 'green', \
    'Coculture ESAM -': 'gold', 'Coculture ESAM +': 'purple'}
# %%
fontSize = 20
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)
for label in set(labels):
    labelIdx = np.where(np.array(labels)==label)
    ux = u[labelIdx,0]
    uy = u[labelIdx,1]
    ax.scatter(ux, uy, s=5, c=label2Color[label], alpha=0.5, label=label)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('ESAM Texture/Morphology')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.title.set_size(      fontSize)
ax.xaxis.label.set_size(fontSize)
ax.yaxis.label.set_size(fontSize)
ax.legend(markerscale=4)
ax.set_yticks([])
ax.set_xticks([])



# ax.scatter(u[:,0], u[:,1], s=5, c=wellColor, alpha=0.5)
# fig.savefig('../results/figs/esamTexture.png', dpi=600)

# plt.scatter(u[:,0], u[:,1], c = wellColor, s= 2, alpha=0.5)
# %%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(normalized_df.values)
plt.scatter(u[:,0], u[:,1], c = kmeans.labels_ , s= 2, alpha=0.5)
# %%