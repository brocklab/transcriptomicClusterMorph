# %%
import sys, importlib
# importlib.reload(sys.modules['cellMorphHelper'])
import pickle
import os
import cv2
import numpy as np
from skimage.io import imread

import matplotlib.pyplot as plt
from cellMorphHelper import getSegmentModel, findFluorescenceColor, interpolatePerimeter, procrustes
import cellMorphHelper
from cellMorph import cellPerims

from skimage import data, measure
from skimage.segmentation import clear_border

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
# %%    
predictor = cellMorphHelper.getSegmentModel('../models/AG2021Split16')
# %%
experiment = 'TJ2201Split16'
print('Starting Experiment: {}'.format(experiment))
imNames = os.listdir(os.path.join('../data',experiment, 'phaseContrast'))

cells = []

c = 0
cellMorphHelper.printProgressBar(0, total = len(imNames), length = 50)
for imName in imNames:
    c+=1
    imBase = cellMorphHelper.getImageBase(imName)
    splitNum = imBase.split('_')[-1]
    well = imBase.split('_')[0]
    if well != 'D2':
        continue
    print(f'Processing {imName} \n')
    im = imread(os.path.join('../data', experiment, 'phaseContrast', imName))

    outputs = predictor(im)['instances'].to("cpu")
    nCells = len(outputs)

    for cellNum in range(nCells):
        mask = outputs[cellNum].pred_masks.numpy()[0]
        mask = clear_border(mask)
        if np.sum(mask)>10:
            cells.append(cellPerims(experiment, imBase, splitNum, mask))

    cellMorphHelper.printProgressBar(c, total = len(imNames), length = 50)

    # Save periodically
    if c % 100 == 0:
        print('\t Saving at ../results/{}ESAMPos.pickle'.format(experiment))
        pickle.dump(cells, open('../results/{}ESAMPos.pickle'.format(experiment), "wb"))

pickle.dump(cells, open('../results/{}ESAMPos.pickle'.format(experiment), "wb"))

# %% Align red and green cells
print('Aligning Perimeters')

cells[0].perimAligned = cells[0].perimInt - np.mean(cells[0].perimInt, axis=0)
referencePerim = cells[0].perimAligned
c = 1
for cell in cells[1:]:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = procrustes(referencePerim, currentPerim, scaling=False)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)

pickle.dump(cells, open('../results/{}ESAMPos.pickle'.format(experiment), "wb"))
