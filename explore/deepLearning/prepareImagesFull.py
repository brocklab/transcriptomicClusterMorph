# %%
import sys
sys.path.append('../')
import os
import numpy as np
import torch
import random
import shutil

from skimage.io import imread
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

import cellMorphHelper
# %%
rebuild_data = True

class esamMono():
    imPath = '../../data/TJ2201/phaseContrast'

    labels = {'esamNegative': 0, 'esamPositive': 1}
    training_data = []

    def make_training_data(self):
        for f in tqdm(os.listdir(self.imPath)):
            well = f.split('_')[1]
            date = cellMorphHelper.convertDate('_'.join(cellMorphHelper.getImageBase(f).split('_')[2:]))
            if date < datetime.datetime(2022, 4, 8, 16, 0):
                continue
            if well == 'E2':
                label = 'esamNegative'
            elif well == 'D2':
                label = 'esamPositive'
            else:
                continue
            path = os.path.join(self.imPath, f)
            img = imread(path)
            self.training_data.append([np.array(img), np.eye(2)[self.labels[label]]])
        np.random.shuffle(self.training_data)
        np.save('../../data/esamMonoSegmented/training_data_fullIms.npy', self.training_data)

if rebuild_data == True:
    esamMonoDat = esamMono()
    esamMonoDat.make_training_data()
# %%
esamMonoData = np.load('../../data/esamMonoSegmented/training_data_fullIms.npy', allow_pickle=True)