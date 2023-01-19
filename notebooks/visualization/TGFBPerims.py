# %%
import sys
sys.path.append('../')
import src.data.cellMorphHelper as cellMorphHelper
import src.data.cellMorph as cellMorph

import os
import pickle
import matplotlib.pyplot as plt
import umap
import numpy as np
# %%
# Plate map image is stored on phone btw
resDir = '../../results/AG2217-ESAM-TGFB'
esamNeg =           pickle.load(open(os.path.join(resDir,'AG2217-ESAM-TGFB-B2.pickle'),"rb"))
TGFB10esamNeg  =    pickle.load(open(os.path.join(resDir,'AG2217-ESAM-TGFB-B4.pickle'),"rb"))
esamPos =           pickle.load(open(os.path.join(resDir,'AG2217-ESAM-TGFB-B9.pickle'),"rb"))
TGFB10esamPos =     pickle.load(open(os.path.join(resDir,'AG2217-ESAM-TGFB-B11.pickle'),"rb"))
coculture =         pickle.load(open(os.path.join(resDir,'AG2217-ESAM-TGFB-B5.pickle'),"rb"))
TGFB10coculture =   pickle.load(open(os.path.join(resDir,'AG2217-ESAM-TGFB-B6.pickle'),"rb"))

cells = esamNeg+TGFB10esamNeg+esamPos+TGFB10esamPos+coculture+TGFB10coculture
# %%
allPerims = np.array([cell.perimAligned.ravel() for cell in cells])
allLabels = ['esamNeg' for x in range(len(esamNeg))]+\
            ['TGFB10esamNeg' for x in range(len(TGFB10esamNeg))]+\
            ['esamPos' for x in range(len(esamPos))]+\
            ['TGFB10esamPos' for x in range(len(TGFB10esamPos))]+\
            ['coculture' for x in range(len(coculture))]+\
            ['TGFB10coculture' for x in range(len(TGFB10coculture))]
# %%
fit = umap.UMAP()
u = fit.fit_transform(allPerims)
# %%
fontSize = 20
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)

colors = ['red', 'green', 'blue', 'purple', 'gold', 'cyan', 'pink']
label2Color = {label: color for label, color in zip(set(allLabels), colors)}

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for label in np.unique(allLabels):
    labelIdx = np.where(np.array(allLabels)==label)
    ux = u[labelIdx,0]
    uy = u[labelIdx,1]
    ax.scatter(ux, uy, s=3, c=label2Color[label], alpha=0.25, label=label)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('ESAM Morphology')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.title.set_size(      fontSize)
ax.xaxis.label.set_size(fontSize)
ax.yaxis.label.set_size(fontSize)
ax.legend(markerscale=4)
ax.set_yticks([])
ax.set_xticks([])
# fig.savefig('../results/figs/esamCoMonoUMAP.png', dpi=600)