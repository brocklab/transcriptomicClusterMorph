# %%
# %%
import cellMorphHelper

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
# %%
predictor = cellMorphHelper.getSegmentModel('../models/AG2021Split16')
# %% Visualize proper order of images
composite = imread('../data/TJ2201/composite/composite_D2_6_2022y04m07d_00h00m.png')
pc = imread('../data/TJ2201/phaseContrast/phaseContrast_D2_6_2022y04m07d_00h00m.png')
# %% 
imPath = '../data/TJ2201Split16/phaseContrast/phaseContrast_D2_6_2022y04m07d_00h00m_4.png'
cellMorphHelper.viewPredictorResult(predictor, imPath)
# %% Connect images
# Images are connected column wise, whereas matplotlib plots are row-wise
# Connect them properly by first creating a dictionary of the correct orientation
# NOTE: Restricted to split-16 images
nIms = 16
nSplit = int(np.sqrt(nIms))
assert nSplit == np.sqrt(nIms)
splitNums = list(range(1,nIms+1))
matplotNums = list(range(1,nIms+1))

c = 1
rowNum = 0
imSplitToMatplot = {}
for row in range(nSplit):

    rowNum += 1
    colNum = rowNum
    for col in range(nSplit):
        imSplitToMatplot[colNum] = c
        colNum += nSplit
        c+=1

# Test by plotting this out
plt.figure()
for imNum in range(1, nIms+1):
    imPath = f'../data/TJ2201Split16/phaseContrast/phaseContrast_D2_6_2022y04m07d_00h00m_{imNum}.png'
    im = imread(imPath)
    plt.subplot(nSplit, nSplit, imSplitToMatplot[imNum])
    plt.imshow(im)
    plt.tick_params(
    axis='both',       
    which='both',      
    bottom=False,      
    top=False,
    left=False,
    labelleft=False,   
    labelbottom=False)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

plt.show()

plt.figure()
imPath = f'../data/TJ2201/phaseContrast/phaseContrast_D2_6_2022y04m07d_00h00m.png'
plt.imshow(imread(imPath))
# %% Actually stitch images
# Stitch columns
columns = []
imNum = 1
for column in range(nSplit):
    rows = []
    for row in range(nSplit):
        imPath = f'../data/TJ2201Split16/phaseContrast/phaseContrast_D2_6_2022y04m07d_00h00m_{imNum}.png'
        im = imread(imPath)
        rows.append(im)
        imNum+=1
    columns.append(np.vstack(rows))
concatIm = np.hstack(columns)
plt.imshow(concatIm)
# %% Incorporate masks
for imNum in range(1, nIms+1):
    imPath = f'../data/TJ2201Split16/phaseContrast/phaseContrast_D2_6_2022y04m07d_00h00m_{imNum}.png'
    im = imread(imPath)
    outputs = predictor(im)['instances'].to("cpu")
    borderOutputs = []
    