# %%
# %%
import pickle
import random
import pandas as pd
import cellMorphHelper
import numpy as np
import datetime

import matplotlib.pyplot as plt
import umap

from skimage.io import imread
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation
from sklearn.cluster import KMeans
# %%
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
esamNeg = [cell for cell in esamNeg if cell.date < datetime.datetime(2022, 4, 8, 16, 0) and cell.color=='red']
# %%
predictor = cellMorphHelper.getSegmentModel('../models/AG2021Split16')
# %%
im = imread(esamNeg[0].composite)
cellMorphHelper.viewPredictorResult(predictor, esamNeg[0].phaseContrast)
# %%
outputs = predictor(im)['instances'].to("cpu")
nCells = len(outputs)

# Append information for each cell
for cellNum in range(nCells):
    mask = outputs[cellNum].pred_masks.numpy()[0]
    # Don't use cells that are on the border for grabbing perimeter information
    # TODO: Merge all images
    maskDilate = binary_dilation(mask)
    maskFinal = clear_border(maskDilate)

    plt.figure(figsize=(10,5))
    plt.subplot(131)
    plt.imshow(mask)
    plt.subplot(132)
    plt.imshow(maskDilate)
    plt.subplot(133)
    plt.imshow(maskFinal)
# %%
cellNum = 3
mask = outputs[cellNum].pred_masks.numpy()[0]
# Don't use cells that are on the border for grabbing perimeter information
# TODO: Merge all images
plt.figure()
mask = clear_border(mask)
plt.imshow(mask)