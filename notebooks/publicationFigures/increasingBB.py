# %% [markdown]
"""
Generates a figure demonstrating increasing bounding boxes for a cell
"""

# %%
from src.data.imageProcessing import bbIncrease
from src.data.fileManagement import splitName2Whole
from src.visualization.trainTestRes import plotTrainingRes

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
homePath = Path('../../')
datasetDictPath = homePath / './data/TJ2201/split16/TJ2201DatasetDictNoBorderFull.npy'
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %% Get single cell
idx = random.randint(0,len(datasetDicts))
# idx = 15902
idx = 15909
print(idx)
c = idx
for seg in datasetDicts[idx:]:
    annotations = seg['annotations']
    nCells = len(seg['annotations'])
    if nCells < 10:
        # print(f'Skipping {c} \t {nCells}')
        c += 1
        continue
    imgPath = str(homePath / Path(*Path(seg['file_name']).parts[2:]))
    imgPath = imgPath.replace('/raw/', '/split16/')

    img = imread(imgPath)

    cell = annotations[2]

    break
print(c)
polyx = cell['segmentation'][0][0::2]
polyy = cell['segmentation'][0][1::2]
bb = cell['bbox']
# plt.imshow(img)
# plt.plot(polyx, polyy, 'b--', linewidth=3)

# Plot only cell
maxRows, maxCols = 150, 150
polygon = list(zip(polyy, polyx))

mask = polygon2mask(img.shape[0:2], polygon)
bb = [int(corner) for corner in bb]
pcCrop = img[bb[1]:bb[3], bb[0]:bb[2]].copy()
maskCrop = mask[bb[1]:bb[3], bb[0]:bb[2]].copy().astype('bool')

pcCrop[~np.dstack((maskCrop,maskCrop,maskCrop))] = 0
pcCrop = torch.tensor(pcCrop[:,:,0])
# %%
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
# %%
# Increase bounding box
imgNameWhole = splitName2Whole(seg['file_name'].split('/')[-1])
imgPathWhole = homePath / 'data/TJ2201/raw/phaseContrast' / imgNameWhole
imgWhole = imread(imgPathWhole)
nIncreases = [0, 25, 65]
increasingBB = {}
num = 1
for nIncrease in nIncreases:
    polyx = cell['segmentation'][0][0::2]
    polyy = cell['segmentation'][0][1::2]
    poly = np.array([polyx, polyy]).T
    imgCrop = bbIncrease(poly, cell['bbox'], seg['file_name'], imgWhole, nIncrease = nIncrease, nIms=16)
    increasingBB[nIncrease] = imgCrop
    print(imgCrop.shape)
# 
fig = plt.figure(constrained_layout=True, figsize=(10, 4))
sfs = fig.subfigures(1, 3)
c = 0
startDict = {
                0:  [1, 1],
                25: [1, 1],
                65: [1, 1]}
factor = 200/322 #200 um/ 322 px
pxIncrease = int(10/factor)
for nIncrease in nIncreases:
    img =increasingBB[nIncrease]
    start = startDict[nIncrease]

    # img[start[0]:start[0]+2, start[1]:(start[1]+pxIncrease)] = 0
    sf = sfs[c]
    ax = sf.add_axes([0, 0, 1, 0.85])
    ax.imshow(img, cmap = 'gray')
    sf.suptitle(f'{nIncrease} px\nIncrease', fontsize=15)
    ax.axis('off')
    c += 1
fig.savefig(homePath / 'figures/publication/results/increasingBBDemonstration.png', dpi=600)
# %%
from tqdm import tqdm
from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode

def convertRecords(datasetDicts):
    newDatasetDicts = []
    for record in tqdm(datasetDicts):
        well = record['file_name'].split('_')[1]
        if well.endswith('10') or well.endswith('11'):
            continue
        record = record.copy()

        for annotation in record['annotations']:
            annotation['bbox'] = BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
        newDatasetDicts.append(record)
    return newDatasetDicts
experiment  = 'TJ2453-436Co'
datasetDicts = load_coco_json(homePath / f'data/{experiment}/{experiment}SegmentationsFiltered.json', '.')
datasetDicts = convertRecords(datasetDicts)
# %%
from src.data.imageProcessing import imSplit
idx = random.randint(0,len(datasetDicts))
# idx = 15902
idx = 18836
c = idx
for seg in datasetDicts[idx:]:
    annotations = seg['annotations']
    nCells = len(seg['annotations'])
    if nCells < 10:
        # print(f'Skipping {c} \t {nCells}')
        c += 1
        continue
    imgPath = homePath / 'data' / Path(*Path(seg['file_name']).parts[2:])

    imgPath = str(imgPath)
    
    imgSplitPath = imgPath.split('_')
    imNum = int(imgSplitPath[-1].split('.png')[0])

    imgPath = '_'.join(imgSplitPath[0:-1])+'.png'

    img = imread(imgPath)

    imgSplit = imSplit(img, nIms = 16)
    img = imgSplit[imNum]
    img = np.array([img, img, img]).transpose([1, 2, 0])

    cell = annotations[2]

    break
print(c)
polyx = cell['segmentation'][0][0::2]
polyy = cell['segmentation'][0][1::2]
bb = cell['bbox']
# plt.imshow(img)
# plt.plot(polyx, polyy, 'b--', linewidth=3)

# Plot only cell
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


# Increase bounding box
imgNameWhole = splitName2Whole(seg['file_name'].split('/')[-1])
imgPathWhole = homePath / f'data/{experiment}/raw/phaseContrast' / imgNameWhole
imgWhole = imread(imgPathWhole)
nIncreases = [25, 55, 75]
increasingBB = {}
num = 1
for nIncrease in nIncreases:
    polyx = cell['segmentation'][0][0::2]
    polyy = cell['segmentation'][0][1::2]
    poly = np.array([polyx, polyy]).T
    imgCrop = bbIncrease(poly, cell['bbox'], seg['file_name'], imgWhole, nIncrease = nIncrease, nIms=16)
    increasingBB[nIncrease] = imgCrop
fig = plt.figure(constrained_layout=True, figsize=(10,3))
sfs = fig.subfigures(1, 3)
c = 0
nIncrease0 = nIncreases[0]
ratio = increasingBB[nIncreases[0]].shape[0]/increasingBB[nIncreases[0]].shape[1]
factor = 200/322 #200 um/ 322 px
pxIncrease = int(30/factor)
startDict = {
                25: 2,
                55: 2,
                75: 2}
for nIncrease in nIncreases:
    start = startDict[nIncrease]
    img =increasingBB[nIncrease]

    # img[start:start+5, start:(start+pxIncrease)] = 0
    sf = sfs[c]
    ax = sf.add_axes([0, 0, 1, 0.85])
    ax.imshow(img, cmap = 'gray')
    d = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    # ax.set_aspect(-d)
    ax.set_box_aspect(ratio)
    sf.suptitle(f'{nIncrease} px\nIncrease', fontsize=15)
    ax.axis('off')
    c += 1

fig.savefig(homePath / 'figures/publication/results/increasingBBDemonstration436.png', dpi=600)

# %%
augmentations = [None, 'blackoutCell', 'stamp']
changingAugmentations = {}
for augmentation in augmentations:
    polyx = cell['segmentation'][0][0::2]
    polyy = cell['segmentation'][0][1::2]
    poly = np.array([polyx, polyy]).T
    imgCrop = bbIncrease(poly, cell['bbox'], seg['file_name'], imgWhole, nIncrease = 25, nIms=16, augmentation=augmentation)
    changingAugmentations[augmentation] = imgCrop
# %%
fig = plt.figure(constrained_layout=True, figsize=(10, 4))
sfs = fig.subfigures(1, 3)
augmentationNameDict = {
    None: 'No Augmentation',
    'blackoutCell': 'No Texture',
    'stamp': 'No Surrounding'
}
c = 0
for augmentation in augmentations:
    sf = sfs[c]
    ax = sf.add_axes([0, 0, 1, 0.9])
    ax.imshow(changingAugmentations[augmentation], cmap = 'gray')
    sf.suptitle(augmentationNameDict[augmentation], fontsize=15)
    ax.axis('off')
    c += 1
fig.savefig(homePath / 'figures/publication/results/augmentationsDemonstration.png', dpi=600)
# %%
idx = 15909
print(idx)
c = idx
nCellsFound = 0
allCells = []
allSegs = []
for seg in datasetDicts[idx:]:
    annotations = seg['annotations']
    nCells = len(seg['annotations'])
    if nCells < 10:
        # print(f'Skipping {c} \t {nCells}')
        c += 1
        continue
    imgPath = str(homePath / Path(*Path(seg['file_name']).parts[1:]))
    imgPath = imgPath.replace('/raw/', '/split16/')
    img = imread(imgPath)

    cell = annotations[2]

    allCells.append(cell)
    allSegs.append(seg)
    if len(allCells) >=3:
        break

# %%
# allImgs = []
# for cell, seg in zip(allCells, allSegs):
#     imgNameWhole = splitName2Whole(seg['file_name'].split('/')[-1])
#     imgPathWhole = homePath / 'data/TJ2201/raw/phaseContrast' / imgNameWhole
#     imgWhole = imread(imgPathWhole)
#     polyx = cell['segmentation'][0][0::2]
#     polyy = cell['segmentation'][0][1::2]
#     poly = np.array([polyx, polyy]).T
#     imgCrop = bbIncrease(poly, cell['bbox'], seg['file_name'], imgWhole, nIncrease = 0, nIms=16, augmentation=None)
#     allImgs.append(imgCrop)

# fig = plt.figure(constrained_layout=True, figsize=(10, 4))
# sfs = fig.subfigures(1, 3)

# c = 0
# for cellImg in allImgs:
#     sf = sfs[c]
#     ax = sf.add_axes([0, 0, 1, 0.9])
#     ax.imshow(cellImg, cmap = 'gray')
#     # sf.suptitle(augmentationNameDict[augmentation], fontsize=15)
#     ax.axis('off')
#     c += 1
# plt.suptitle('Example Inputs', fontsize = 20)
# fig.savefig(homePath / 'figures/example0IncreaseInputs.png', dpi=600)
# %%