# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib

from skimage.io import imread
from skimage.color import label2rgb
from skimage.measure import label

import cellMorphHelper
# %% Full segmentation
imFull = imread('../data/TJ2201/segmentedIms/composite_D2_4_2022y04m08d_08h00m.png')
imSeg = np.load('../data/TJ2201/segmentedIms/composite_D2_4_2022y04m08d_08h00m_seg.npy', allow_pickle=True)
imSeg = imSeg.item()

label_img = label(imSeg['masks'])
imgOverlay = label2rgb(label_img, imFull[:,:,0:3])

resFactor = 1408/1040

fontSize = 20
fig, ax = plt.subplots()
imSizeX = 20
fig.set_size_inches(imSizeX, imSizeX*resFactor)
ax.imshow(imgOverlay)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_yticks([])
ax.set_xticks([])

fig.savefig('../results/figs/exampleFullSeg.png', dpi=600)

# %% Example segmentations
predictorTJ2201Split = cellMorphHelper.getSegmentModel('../output/TJ2201Split16')
predictorAG2021 = cellMorphHelper.getSegmentModel('../output/AG2021Split16')
predictorTJ2201 = cellMorphHelper.getSegmentModel('../output/TJ2201')

# %%
imNum = 9
im = imread(f'../data/TJ2201Split16/phaseContrast/phaseContrast_E2_4_2022y04m07d_16h00m_{imNum}.png')
cellMorphHelper.viewPredictorResult(predictorTJ2201Split, f'../data/TJ2201Split16/phaseContrast/phaseContrast_E2_4_2022y04m07d_16h00m_{imNum}.png')
cellMorphHelper.viewPredictorResult(predictorAG2021, f'../data/TJ2201Split16/phaseContrast/phaseContrast_E2_4_2022y04m07d_16h00m_{imNum}.png')
# %% Perimeter extraction
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))

# Constrain to low confluency
esamNeg = [cell for cell in esamNeg if cell.date < datetime.datetime(2022, 4, 8, 16, 0)]
# Filter color
esamNeg = [cell for cell in esamNeg if cell.color=='red']
# Filter borders
esamNeg = [cell for cell in esamNeg if cellMorphHelper.clearEdgeCells(cell) == 1]

# %% Plot perimeters
matplotlib.rcParams.update({'font.size': 19})
cell = esamNeg[2]
markerSize = 3
markerType = 'o'

perim = cell.perimeter
perimInt = cell.perimInt
perimAligned = cell.perimAligned

plt.figure(figsize = (14,10/3))
plt.subplot(131)
plt.scatter(perim[:,0], perim[:,1], s = markerSize, marker = markerType)
plt.title('Original')

plt.subplot(132)
plt.scatter(perimInt[:,0], perimInt[:,1], s = markerSize, marker = markerType)
plt.title('Interpolated')

plt.subplot(133)
plt.scatter(perimAligned[:,0], perimAligned[:,1], s = markerSize, marker = markerType)
plt.title('Aligned/Rotated')

plt.savefig('../results/figs/perimeterProcess.png', dpi=600)
plt.show()
# %% Features
featureRes = pickle.load(open('../results/allFeaturesTJ2201.pickle',"rb"))

print(f'{len(featureRes[1][0])} total features')
# %%
