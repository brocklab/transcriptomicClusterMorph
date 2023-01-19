#!/usr/bin/python3
# %%
import sys, importlib
# importlib.reload(sys.modules['cellMorphHelper'])
import pickle
import os
import cv2
import numpy as np
from skimage.io import imread

import matplotlib.pyplot as plt
from cellMorph import cellPerims
import cellMorphHelper

from skimage import data, measure
from skimage.segmentation import clear_border

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# %% Functions
def alignPerimeters(cells: list):
    """
    Aligns a list of cells of class cellPerims
    Inputs:
    cells: A list of instances of cellPerims
    Ouputs:
    List with the instance variable perimAligned as an interpolated perimeter aligned
    to the first instance in list.
    """
    # Create reference perimeter from first 100 cells
    referencePerimX = []
    referencePerimY = []
    for cell in cells[0:1]:
        # Center perimeter
        originPerim = cell.perimInt.copy() - np.mean(cell.perimInt.copy(), axis=0)
        referencePerimX.append(originPerim[:,0])
        referencePerimY.append(originPerim[:,1])
    # Average perimeters
    referencePerim = np.array([ np.mean(np.array(referencePerimX), axis=0), \
                                np.mean(np.array(referencePerimY), axis=0)]).T

    # Align all cells to the reference perimeter
    c = 1
    for cell in cells:
        currentPerim = cell.perimInt
        
        # Perform procrustes to align orientation (not scaled by size)
        refPerim2, currentPerim2, disparity = cellMorphHelper.procrustes(referencePerim, currentPerim, scaling=False)

        # Put cell centered at origin
        cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
    return cells
# %%
predictor = cellMorphHelper.getSegmentModel('../models/TJ2201Split16')
# %% Find masks for experiment
experiment = sys.argv[1]

# Check if split images exist
if not os.path.isdir(os.path.join('../data', experiment+'Split16')):
    print('Experiment is unsplit, splitting...')
    cellMorphHelper.splitExpIms(experiment)
else:
    print('Experiment has been split')

# Validate experiment
cellMorphHelper.validateExperimentData(experiment)
experiment = experiment+'Split16'

# Make a results directory
resDir = f'../results/{experiment}'
os.makedirs(resDir, exist_ok=True)
# %%
print('Starting Experiment: {}'.format(experiment))
ims = os.listdir(os.path.join('../data',experiment, 'phaseContrast'))
imbases = ['_'.join(im.split('.')[0].split('_')[1:-1]) for im in ims]
splitNums = [int(im.split('.')[0].split('_')[-1]) for im in ims]

cells = []

c = 1
# Go through each image
for imbase, splitNum in zip(imbases, splitNums):
    fname = f'phaseContrast_{imbase}_{str(splitNum)}.png'
    imPath = os.path.join('../data',experiment,'phaseContrast',fname)
    im = imread(imPath)
    outputs = predictor(im)['instances'].to("cpu")
    nCells = len(outputs)

    # Append information for each cell
    for cellNum in range(nCells):
        mask = outputs[cellNum].pred_masks.numpy()[0]
        # Don't use cells that are on the border for grabbing perimeter information
        # TODO: Merge all images
        mask = clear_border(mask)
        # Check that cell is not just some artifact
        if np.sum(mask)>10:
            cells.append(cellPerims(experiment, imbase, splitNum, mask))
    c+=1

    # Save periodically
    if c % 100 == 0:
        print(f'Saving at ../results/{experiment}.pickle')
        pickle.dump(cells, open(os.path.join(resDir, f"{experiment}.pickle"), "wb"))
pickle.dump(cells, open(os.path.join(resDir, f'{experiment}.pickle'), "wb"))
# %%
cells = pickle.load(open(os.path.join(resDir, f'{experiment}.pickle'),"rb"))
# Align cells
print('Aligning Cells')
cells = alignPerimeters(cells)

# Split into wells and save results

# Get unique wells
wells = []
for cell in cells:
    well = cell.imageBase.split('_')[0]
    wells.append(well)

# Split cells associated with well into dictionary
wellDict = {well: [] for well in np.unique(wells)}

for cell in cells:
    well = cell.imageBase.split('_')[0]
    wellDict[well].append(cell)


for well in wellDict.keys():
    saveFile = os.path.join(resDir, f'{experiment}-{well}.pickle')
    print(saveFile)
    pickle.dump(wellDict[well], open(saveFile, "wb"))
# %%