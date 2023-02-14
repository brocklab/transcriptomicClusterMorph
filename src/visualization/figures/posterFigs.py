# %%
from src.data.fileManagement import convertDate

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from skimage.io import imread
from skimage.measure import label
from skimage.color import label2rgb

from skimage.draw import polygon2mask
from skimage.draw import polygon
# %%
datasetDicts = np.load('../data/TJ2201/split16/TJ2201DatasetDict.npy', allow_pickle=True)
# %%
# Find images with only one cell
oneCell = []
for seg in datasetDicts:
    if len(seg['annotations']) > 10:
        oneCell.append(seg)

# %%
idx = 2 # '../data/TJ2201/split16/phaseContrast/phaseContrast_E7_6_2022y04m07d_20h00m_11.png'
sz = 4
for idx in range(2,10):
    seg = oneCell[idx]

    filePath = Path(seg['file_name'])
    # Get rid of relative portions
    filePath =  '../' / Path(*filePath.parts[2:])
    img = imread(filePath)
    plt.figure(figsize = (sz, sz))
    plt.imshow(imread(str(filePath).replace('phaseContrast', 'composite')))
    plt.axis('off')
    plt.savefig(f'../figures/pipelineFigure/whole{idx}.png', dpi=500)

    # Make mask
    plt.figure(figsize = (sz, sz))
    plt.imshow(img)
    for annotation in seg['annotations']:
        polyx = annotation['segmentation'][0][::2]
        polyy = annotation['segmentation'][0][1::2]

        poly = np.array([[y,x] for x,y in zip(polyx, polyy)])
        plt.plot(polyx, polyy, linewidth = 3)
    plt.axis('off')
    plt.savefig(f'../figures/pipelineFigure/segmented{idx}.png', dpi=500)


    mask = np.zeros(np.shape(img))
    for annotation in seg['annotations']:
        polyx = annotation['segmentation'][0][::2]
        polyy = annotation['segmentation'][0][1::2]

        poly = np.array([[y,x] for x,y in zip(polyx, polyy)])
        # plt.plot(polyx, polyy)


        rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
        if annotation['category_id'] == 1:
            mask[rr, cc, :] = (255,0,0)
        else:
            mask[rr, cc, :] = (0,255,0)

    alphaVal = 0.8
    combined = np.ubyte(alphaVal*img + (1-alphaVal)*mask)
    plt.figure(figsize = (sz, sz))
    plt.imshow(combined)
    plt.axis('off')

    plt.savefig(f'../figures/pipelineFigure/combined{idx}.png', dpi=500)
# %%
dateCt = {}
allWells = []
for seg in datasetDicts:
    fileName = Path(seg['file_name']).parts[-1]
    well = fileName.split('_')[1]
    allWells.append(well)
    if well != 'B7':
        continue
    dateIncucyte = '_'.join(fileName.split('_')[3:5])
    date = convertDate(dateIncucyte)
    if date not in dateCt.keys():
        dateCt[date] = [0, 0]
    for cell in seg['annotations']:
        dateCt[date][cell['category_id']] += 1

# %%
import matplotlib.dates as mdates

plt.rcParams.update({'font.size': 18})
dateCtDf = pd.DataFrame(dateCt).transpose()
dateCtDf['date'] = pd.to_datetime(dateCtDf.index)
plt.rc('axes', axisbelow=True)
plt.figure(figsize=(7.5,6))
# plt.locator_params(axis='x',nbins = 3)
plt.grid()
plt.plot(dateCtDf['date'], dateCtDf[0], 'o', color = 'green', label='Subpopulation 1', markersize=9)
plt.plot(dateCtDf['date'], dateCtDf[1], 'o', color = 'red', label='Subpopulation 2', markersize=9)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.HourLocator(interval=22))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.xticks(rotation=0)
plt.xlabel('Date')
plt.ylim([0,700])
plt.ylabel('Number of Cells')
plt.legend(loc='upper left')

plt.savefig('../figures/pipelineFigure/growthPlot.png', dpi=600)
# %%