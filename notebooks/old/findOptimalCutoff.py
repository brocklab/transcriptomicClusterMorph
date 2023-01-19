# %% [markdown]
"""
# A notebook to track the optimal time for segmentation. 

When cells grow too confluent, they are difficult to segment.
"""
# %%
# %%
import cellMorph
import cellMorphHelper

import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from skimage.io import imread
from skimage.measure import label
from skimage.color import label2rgb

# %%
def labelImage(cells, cell, imageBases, date):
    # Find where other cells have been identified
    imageBase = cell.imageBase
    splitNum = cell.splitNum
    cellIdx = np.where(imageBases == f'{imageBase}_{splitNum}')[0]
    # Collect their perimeters
    perims = []
    masks = np.zeros(np.shape(cells[0].mask))
    c = 1
    for idx in cellIdx:
        perims.append(cells[idx].perimeter)
        mask = cells[idx].mask
        masks[mask>0] = c
        c+=1
    # Plot image
    composite = imread(cells[cellIdx[0]].composite)

    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.imshow(composite)
    for perim in perims:
        plt.plot(perim[:,1], perim[:,0])
    plt.title(date)
    plt.tick_params(
        axis='both',       
        which='both',      
        bottom=False,      
        top=False,
        left=False,
        labelleft=False,   
        labelbottom=False)

    plt.subplot(2,1,2)
    label_image = label(masks)
    overlay = label2rgb(label_image, image=imread(cell.phaseContrast),bg_label=0)
    plt.imshow(overlay)
    # plt.tick_params(
    #     axis='both',       
    #     which='both',      
    #     bottom=False,      
    #     top=False,
    #     left=False,
    #     labelleft=False,   
    #     labelbottom=False)

# labelImage(cells, cells[1000], allImageBases)
# %%
print('Loading cell data, hold on...')
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
esamPos = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))
# %%
esamNeg = [cell for cell in esamNeg if cell.color=='red']
esamPos = [cell for cell in esamPos if cell.color=='green']

cells = esamNeg+esamPos
imEsamNeg = [f'{cell.imageBase}_{cell.splitNum}' for cell in (esamNeg)]
imEsamPos = [f'{cell.imageBase}_{cell.splitNum}' for cell in (esamPos)]

allImageBases = np.array(imEsamNeg+imEsamPos)
# %%
imEsamNeg = [f'{cell.imageBase}_{cell.splitNum}' for cell in (esamNeg)]
imEsamPos = [f'{cell.imageBase}_{cell.splitNum}' for cell in (esamPos)]

allImageBases = np.array(imEsamNeg+imEsamPos)
allDates = np.array([cell.date for cell in esamNeg+esamPos])
dates = np.unique(allDates)
# %% Examine Dates Across 
dates.sort()
# Choose relevant dates
dateIdx = list(np.linspace(0, len(dates)-1, 8).astype("int"))

for date in dateIdx:
    imIdx = np.where(allDates==dates[date])[0]
    imIdx = imIdx[random.randint(0,len(imIdx))]
    plt.figure()
    labelImage(cells, cells[imIdx], allImageBases, str(allDates[imIdx]))
# %%
predictorComposite = cellMorphHelper.getSegmentModel('../models/AG2021Split16Composite')
predictorPhaseContrast = cellMorphHelper.getSegmentModel('../models/AG2021Split16')

# %%
plt.figure()
cellMorphHelper.viewPredictorResult(predictorComposite, cells[imIdx].composite)
plt.figure()
cellMorphHelper.viewPredictorResult(predictorPhaseContrast, cells[imIdx].phaseContrast)
