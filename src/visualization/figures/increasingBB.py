# %% [markdown]
"""
Generates a figure demonstrating increasing bounding boxes for a cell
"""

# %%
from src.data.imageProcessing import bbIncrease
from src.data.fileManagement import splitName2Whole

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random 

from skimage.io import imread
from skimage.draw import polygon2mask
from skimage.transform import rescale, resize

import torch
import torch.nn.functional as F
# %%
homePath = Path('../../../')
datasetDictPath = homePath / './data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %% Get single cell
idx = 516
c = idx
for seg in datasetDicts[idx:]:
    annotations = seg['annotations']
    nCells = len(seg['annotations'])
    if nCells < 10:
        print(f'Skipping {c} \t {nCells}')
        c += 1
        continue
    imgPath = homePath / Path(*Path(seg['file_name']).parts[2:])

    img = imread(imgPath)

    cell = annotations[2]

    break
print(c)
polyx = cell['segmentation'][0][0::2]
polyy = cell['segmentation'][0][1::2]
bb = cell['bbox']
plt.imshow(img)
plt.plot(polyx, polyy, 'b--', linewidth=3)
# %% Plot only cell
maxRows, maxCols = 150, 150
polygon = list(zip(polyy, polyx))

mask = polygon2mask(img.shape[0:2], polygon)
bb = [int(corner) for corner in bb]
pcCrop = img[bb[1]:bb[3], bb[0]:bb[2]].copy()
maskCrop = mask[bb[1]:bb[3], bb[0]:bb[2]].copy().astype('bool')

pcCrop[~np.dstack((maskCrop,maskCrop,maskCrop))] = 0
pcCrop = torch.tensor(pcCrop[:,:,0])

# Keep aspect ratio and scale down data to be maxSize x maxSize (should be rare)

if pcCrop.shape[0]>maxRows:
    pcCrop = rescale(pcCrop, maxRows/pcCrop.shape[0])
if pcCrop.shape[1]>maxCols:
    pcCrop = rescale(pcCrop, maxRows/pcCrop.shape[1])

# Now pad out the amount to make it maxSize x maxSize
diffRows = int((maxRows - pcCrop.shape[0])/2)+1
diffCols = int((maxCols - pcCrop.shape[1])/2)
pcCrop = F.pad(torch.tensor(pcCrop), pad=(diffCols, diffCols, diffRows, diffRows)).numpy()
# Resize in case the difference was not actually an integer
pcCropFull = resize(pcCrop, (maxRows, maxCols))


# %% Increase bounding box
imgNameWhole = splitName2Whole(seg['file_name'].split('/')[-1])
imgPathWhole = homePath / 'data/TJ2201/raw/phaseContrast' / imgNameWhole
imgWhole = imread(imgPathWhole)
nIncreases = [0, 45, 65]
increasingBB = {}
for nIncrease in nIncreases:
    imgCrop = bbIncrease(cell['segmentation'][0], cell['bbox'], seg['file_name'], imgWhole, nIncrease = nIncrease)
    increasingBB[nIncrease] = imgCrop
# %%
plt.figure(figsize=(10,10))

plt.subplot(221)
plt.imshow(pcCropFull, cmap='gray')
plt.title('No bounding box', fontsize=15)
plt.axis('off')

c = 2
for nIncrease in nIncreases:
    plt.subplot(2,2,c)
    plt.imshow(increasingBB[nIncrease], cmap='gray')
    plt.title(f'BB Increase = {nIncrease} px', fontsize=15)
    plt.axis('off')
    c += 1

plt.savefig(homePath / 'figures/increasingBBDemonstration.png', dpi=600)
# %%ß