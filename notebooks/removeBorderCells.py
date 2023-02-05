# %%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation
# %%
datasetDicts = np.load('../data/TJ2201/split16/TJ2201DatasetDict.npy', allow_pickle=True)
# %%
breakOn = 0
ddNoBorder = []
for seg in tqdm(datasetDicts):
    segNoBorder = seg.copy()
    segNoBorder['annotations'] = []
    for i, annotation in enumerate(seg['annotations']):
        mask = np.zeros([seg['height'], seg['width']])
        polyx = annotation['segmentation'][0][::2]
        polyy = annotation['segmentation'][0][1::2]

        poly = np.array([[y,x] for x,y in zip(polyx, polyy)])
        rr, cc = polygon(poly[:, 0], poly[:, 1], mask.shape)

        mask[rr, cc] = [1]
        mask = binary_dilation(mask)

        maskCleared = clear_border(mask)

        if np.sum(maskCleared) != 0:
            segNoBorder['annotations'].append(annotation)
    ddNoBorder.append(segNoBorder)
    

nCellsInit, nCellsFilter = 0, 0
for segInit, segFilter in zip(datasetDicts, ddNoBorder):
    nCellsInit += len(segInit['annotations'])
    nCellsFilter += len(segFilter['annotations'])

print(f'Filtered from {nCellsInit} cells to {nCellsFilter}')
np.save('../data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy', ddNoBorder)
# %%

idx = 0

seg = ddNoBorder[idx]

