# %%
# %%
import pickle
import umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import src.data.cellMorphHelper as cellMorphHelper
# %%
featureRes = pickle.load(open('../results/allFeaturesTJ2201.pickle',"rb"))
colors = pickle.load(open('../results/TJ2201allColors.pickle',"rb"))
# %%
dates = np.array([cell['date'] for cell in featureRes[2]])
features = pd.DataFrame(featureRes[0])
features = features.dropna(axis='columns')

df = features.copy()
normalized_df=(df-df.min())/(df.max()-df.min())
normalized_df = normalized_df.dropna(axis='columns')
# %%
fit = umap.UMAP()
u = fit.fit_transform(normalized_df.values)
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
import matplotlib.dates as mdates
matplotDates = mdates.date2num(dates)
fontSize = 20
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)
# for label in set(labels):
for label in ['Monoculture ESAM +', 'Monoculture ESAM -']:
    labelIdx = np.where(np.array(labels)==label)
    ux = u[labelIdx,0]
    uy = u[labelIdx,1]
    labelDates = matplotDates[labelIdx]
    if '+' in label:
        cmap = 'Greens'
    elif '-' in label:
        cmap = 'Reds'
    ax.scatter(ux, uy, s=5, c=labelDates, alpha=0.5, label=label, cmap=cmap)
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
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig('../results/figs/esamTextureTime.png', dpi=600)
# fig.colorbar()
# %%
