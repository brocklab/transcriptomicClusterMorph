# %%
import numpy as np
import os
import random

from skimage.io import imread

from torch.utils.data import Dataset, DataLoader
import torch
# %%
class esamSplit(Dataset):

    def __init__(self, imDir, phenoWellDict, n, phase, seed=1234):
        self.imDir = imDir
        self.seed = seed
        self.n = n
        self.phase = phase
        self.phenoWellDict = phenoWellDict
        # Class part
        files = os.listdir(imDir)
        random.seed(seed)
        random.shuffle(files)

        # Get finalized number of images
        phenoCount = {0:0, 1:0}
        finalIms = []
        for file in files:
            well = file.split('_')[1]
            
            if well in phenoWellDict.keys() and phenoCount[phenoWellDict[well]]<n:
                finalIms.append(file)
                phenoCount[phenoWellDict[well]] += 1

        if phase == 'train':
            n = int(len(finalIms)*0.9)
            finalIms = finalIms[0:n]
        elif phase == 'test':
            n = int(len(finalIms)*0.9)
            finalIms = finalIms[n:]
        self.finalIms = finalIms
    def __len__(self):
        return len(self.finalIms)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgName = self.finalIms[idx]
        imgPath = os.path.join(self.imDir, imgName)
        img = imread(imgPath)

        return img

# %%
monoPos = ['B2','B3','B4','B5','B6','C2','C3','C4','C5','C6','D2','D3','D4','D5','D6']
monoNeg = ['E2','E3','E4','E5','E6','F2','F3','F4','F5','F6','G2','G3','G4','G5','G6']

phenoWellDict = {}
for well in monoNeg:
    phenoWellDict[well] = 0
for well in monoPos:
    phenoWellDict[well] = 1
imDir = '../../data/TJ2201Split16/phaseContrast'
seed = 1234
n = 10000

train_dataset = esamSplit(imDir, phenoWellDict, n, phase='train')
train_loader = DataLoader(train_dataset, batch_size=2)
# %%
dataiter = iter(train_loader)
image, label = dataiter.next()
# %%
