# %%
import os
import numpy as np
import torch
import random
import shutil

from skimage.io import imread
import matplotlib.pyplot as plt
# %matplotlib inline
from tqdm import tqdm
# %% Make test data
# dataDir = '../../data/esamMonoSegmented/'
# testingDir = os.path.join(dataDir, 'testing')
# if not os.path.isfile(testingDir):
#     esamPosDir = os.path.join(dataDir, 'esamPositive')
#     esamNegDir = os.path.join(dataDir, 'esamNegative')
#     esamPosFiles = os.listdir(esamPosDir)
#     esamNegFiles = os.listdir(esamNegDir)
#     random.seed(1234)
#     random.shuffle(esamPosFiles)
#     random.shuffle(esamNegFiles)
#     nMove = int(len(esamPosFiles)*.25)

#     for esamPosFile, esamNegFile in zip(esamPosFiles[0:nMove], esamNegFiles[0:nMove]):
#         sh
# %%
rebuild_data = True

class esamMono():
    esamNegative = '../../data/esamMonoSegmented/esamNegative'
    esamPositive = '../../data/esamMonoSegmented/esamPositive'

    labels = {esamNegative: 0, esamPositive: 1}
    training_data = []

    def make_training_data(self):
        for label in self.labels:
            print(label)
            for f in tqdm(os.listdir(label)):
                path = os.path.join(label, f)
                img = imread(path)
                self.training_data.append([np.array(img), np.eye(2)[self.labels[label]]])
        np.random.shuffle(self.training_data)
        np.save('../../data/esamMonoSegmented/training_data.npy', self.training_data)

if rebuild_data == True:
    esamMonoDat = esamMono()
    esamMonoDat.make_training_data()

# %%
x = imread('../../data/esamMonoSegmented/esamNegative/E2_9_2022y04m06d_12h00m_3-17433.png')
plt.imshow(x)
