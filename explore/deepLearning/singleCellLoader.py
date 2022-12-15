# %% [markdown]
"""
This is a script making a data loader using the outline information 
(not loading individual images)
"""
# %%
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.draw import rectangle

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
# %%
class singleCellCrop(Dataset):
    """
    Dataloader class for cropping out a cell from an image
    """

    def __init__(self, datasetDicts, dataPath, phase, randomSeed = 1234):
        self.dataPath = dataPath
        self.phase = phase
        self.seed = randomSeed
        self.balance(datasetDicts)
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        pass
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        imgName = self.imgPaths[idx]
        label = self.phenotypes[idx]
        fullPath = os.path.join()

    def balance(self, datasetDicts):
        """
        Balances the data so that there are equal inputs. 

        Input: 
            - datasetDicts, detectron2 format for segmentations
        Output:
            - segmentation, numpy array of nx2
            - phenotypes, list of encoded phenotypes
            - imgPaths, list of image names
        """
        # Reformat dataset dict to most relevant information
        segmentations, phenotypes, imgPaths, bbs = [], [], [], []
        # Note there is a lot of repeats for images but this is much cleaner
        for img in datasetDicts:
            path = img['file_name'].split('/')[-1]
            for annotation in img['annotations']:
                segmentations.append(np.array(annotation['segmentation'][0]))
                phenotypes.append(annotation['category_id'])
                imgPaths.append(path)
                bbs.append([int(corner) for corner in annotation['bbox']])
        # Balance dataset
        uniquePheno, cts = np.unique(phenotypes, return_counts=True)
        maxAmt = min(cts)

        segmentations, phenotypes, imgPaths, bbs = self.shuffleLists([segmentations, phenotypes, imgPaths, bbs], self.seed)
        uniqueIdx = []
        for pheno in uniquePheno:
            idx = list(np.where(phenotypes == pheno)[0][0:maxAmt])
            uniqueIdx += idx
        random.seed(self.seed)
        random.shuffle(uniqueIdx)
        
        # Create train/test split
        n = int(0.9*len(uniqueIdx))
        if self.phase == 'train':
            uniqueIdx = uniqueIdx[0:n]
        else:
            uniqueIdx = uniqueIdx[n:]

        # Get finalized amts
        self.segmentations = np.array([np.reshape(seg, (int(len(seg)/2), 2)) for seg in segmentations[uniqueIdx]], dtype='object')
        self.phenotypes = phenotypes[uniqueIdx]
        self.imgPaths = imgPaths[uniqueIdx]
        self.bbs = bbs[uniqueIdx]

    @staticmethod
    def shuffleLists(l, seed=1234):
        random.seed(seed)
        l = list(zip(*l))
        random.shuffle(l)

        return [np.array(itm, dtype='object') for itm in list(zip(*l))]
# %%
datasetDicts = np.load(f'./TJ2201DatasetDict.npy', allow_pickle=True)
# %%
x = singleCellCrop(datasetDicts, dataPath = '', phase='train')
    # return list(zip(*l))
# %%
# Reformat dataset dict to most relevant information
segmentations, phenotypes, imgPaths, bbs = [], [], [], []
# Note there is a lot of repeats for images but this is much cleaner
for img in datasetDicts:
    path = img['file_name'].split('/')[-1]
    for annotation in img['annotations']:
        segmentations.append(np.array(annotation['segmentation'][0]))
        bbs.append([int(corner) for corner in annotation['bbox']])
        phenotypes.append(annotation['category_id'])
        imgPaths.append(path)
# Balance dataset
uniquePheno, cts = np.unique(phenotypes, return_counts=True)
maxAmt = min(cts)

segmentations, phenotypes, imgPaths, bbs = singleCellCrop.shuffleLists([segmentations, phenotypes, imgPaths, bbs])
uniqueIdx = []
for pheno in uniquePheno:
    idx = list(np.where(phenotypes == pheno)[0][0:maxAmt])
    uniqueIdx += idx
random.shuffle(uniqueIdx)
# Get finalized amts
segmentations =  np.array([np.reshape(seg, (int(len(seg)/2), 2)) for seg in segmentations[uniqueIdx]])
bbs = bbs[uniqueIdx]
phenotypes = phenotypes[uniqueIdx]
imgPaths = imgPaths[uniqueIdx]



# %%
sizes = []
experiment = 'TJ2201'
maxSize = 150
for n in range(len(segmentations)):
    dataPath = f'../../data/{experiment}/{experiment}Split16/phaseContrast'
    img = imread(os.path.join(dataPath, imgPaths[n]))
    seg = segmentations[n]
    bb = bbs[n]
    nIncrease = 10
    colMin, rowMin, colMax, rowMax = bb
    rowMin -= nIncrease
    rowMax += nIncrease
    colMin -= nIncrease
    colMax += nIncrease

    # Logic checks
    if rowMin <= 0:
        rowMin = 0
    if rowMax > img.shape[0]:
        rowMax = img.shape[0]
    if colMin <= 0:
        colMin = 0
    if colMax >= img.shape[1]:
        colMax = img.shape[1]

    bbIncrease = [colMin, rowMin, colMax, rowMax]
    cropSmall = img[bb[1]:bb[3], bb[0]:bb[2]]
    imgCrop = img[bbIncrease[1]:bbIncrease[3], bbIncrease[0]:bbIncrease[2]]
    # plt.subplot(121)
    # plt.imshow(cropSmall)
    # plt.subplot(122)
    # plt.imshow(imgCrop)

    padding = np.zeros([150,150,3])

    maxRows, maxCols = maxSize, maxSize

    diffRows = int((maxRows - imgCrop.shape[0])/2)+1
    diffCols = int((maxCols - imgCrop.shape[1])/2)
    pcCrop = F.pad(torch.tensor(imgCrop[:,:,0]), pad=(diffCols, diffCols, diffRows, diffRows)).numpy()
    # plt.imshow(pcCrop, cmap='gray')
    sizes
# Resize in case the difference was not actually an integer