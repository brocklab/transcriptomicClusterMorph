# [markdown]
"""
# Making Training/Testing Data
Now that a suitable segmentation model has been developed, we should save individual images
of cells. 

In this notebook, I will load the model, apply it to every split image of monoculture
and save the segmentation. 
# TODO: Finalize after 
"""

#
from src.data import imageProcessing, fileManagement
from src.models import modelTools
import time
import pickle
import os
import shutil

import numpy as np
import random
from tqdm import tqdm
from skimage.io import imread

import datetime

from skimage import measure
from skimage.segmentation import clear_border

from detectron2.structures import BoxMode
#
def replaceDatasetDict(dataPath: str, experiment: str):
    """
    Replaces dataset dictionary with largest dictionary to reduce analyzing the same images repeatedly

    dataPath: String path to find datasetDicts
    experiment: experiment associated with segmentations
    """

    fullDict = os.path.join(dataPath, f'{experiment}DatasetDict.npy')
    modDict0 = os.path.join(dataPath, f'{experiment}DatasetDict-0.npy')
    modDict1 = os.path.join(dataPath, f'{experiment}DatasetDict-1.npy')

    for datasetPath in [fullDict, modDict0, modDict1]:
        if not os.path.exists(datasetPath):
            return

    fullDictSize = os.path.getsize(fullDict)
    modDict0Size = os.path.getsize(modDict0)
    modDict1Size = os.path.getsize(modDict1)

    # for sizes in [fullDictSize, modDict0Size, modDict1Size]:
    if modDict0Size > modDict1Size:
        modDict = modDict0
        modDictSize = modDict0Size
    else:
        modDict = modDict1
        modDictSize = modDict1Size

    if fullDictSize < modDictSize:
        print('Replacing with mod dict')
        shutil.move(modDict, fullDict)
    else:
        print('Keeping dictionary')
        shutil.move(fullDict, modDict)


def segmentExperiment(dataPath: str, imgBases: list, phenoDict: dict, experiment: str, predictor):
    """
    segmentExperiment gathers all segmentations for an experiment

    Inputs:
        - dataPath: Location of data
        - imgBases: The image bases specifying well, date, etc. of image
        - phenoDict: Connects fluorescence to encoded label
        - datasetDicts: Existing segmentations in Detectron format
        - experiment: Experiment from which images were gathered
        - predictor: Model trained for segmentation
    Outputs:
        - saved datasetDict
    """
    replaceDatasetDict(dataPath, experiment)
    then = time.time()
    # Load the dataset
    datasetDictPath = os.path.join(dataPath, f'{experiment}DatasetDict.npy')
    if os.path.isfile(datasetDictPath):
        datasetDicts = list(np.load(datasetDictPath, allow_pickle=True))
        idx = max([img['image_id'] for img in datasetDicts])+1
    else:
        datasetDicts = []
        idx = 0

    if len(datasetDicts)>0:
        # Replace data path in case folder structure changes, etc.
        for file in datasetDicts:
            fileName = os.path.basename(file['file_name'])
            if 'phaseContrast' in fileName:
                fileName = os.path.join(dataPath, 'phaseContrast', fileName)
            file['file_name'] = fileName
            if not os.path.isfile(fileName):
                raise FileNotFoundError(f'{fileName} was not found, verify working directory')

        processedFiles = [img['file_name'] for img in datasetDicts]
        idx = max([img['image_id'] for img in datasetDicts])+1
    else:
        processedFiles = []
        idx = 0

    saveIdx = 0
    for imgBase in tqdm(imgBases, leave=True):
        # Grab image

        pcFile =        os.path.join('phaseContrast',f'phaseContrast_{imgBase}.png')
        compositeFile = os.path.join('composite',f'composite_{imgBase}.png')
        pcFileFull = os.path.join(dataPath, pcFile)
        compositeFileFull = os.path.join(dataPath, compositeFile)

        if pcFileFull in processedFiles:
            continue

        pcImg = imread(pcFileFull)
        compositeImg = imread(compositeFileFull)

        outputs = predictor(pcImg)['instances'].to("cpu")
        nCells = len(outputs)


        # Go through each cell in each cropped image
        record = {}
        record['file_name'] = pcFileFull
        record['image_id'] = idx
        record['height'] = pcImg.shape[0]
        record['width'] =  pcImg.shape[1]

        cells = []
        # Get segmentation outlines
        for cellNum in range(nCells):
            mask = outputs[cellNum].pred_masks.numpy()[0]
            color = imageProcessing.findFluorescenceColor(compositeImg, mask)
            if color not in ['red', 'green']:
                continue
            contours = measure.find_contours(mask, .5)
            fullContour = np.vstack(contours)

            px = fullContour[:,1]
            py = fullContour[:,0]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            
            bbox = list(outputs[cellNum].pred_boxes.tensor.numpy()[0])

            cell = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": phenoDict[color],
            }

            cells.append(cell)
        record["annotations"] = cells
        datasetDicts.append(record)
        
        idx += 1

        # Alternate saving so that if saving gets interrupted we lose < 1000 images
        if idx % 1000 == 0:
            modIdx = saveIdx % 2
            datasetDictPath = os.path.join(dataPath, f'{experiment}DatasetDict-{modIdx}.npy')
            picklePath = os.path.join(dataPath, f'{experiment}DatasetDict-{modIdx}.pickle')
            print(f'Saving {len(datasetDicts)} images')
            np.save(datasetDictPath, datasetDicts)
            # pickle.dump(datasetDicts, open(picklePath, "wb"))
            print('Done saving')
            saveIdx += 1

        # If time running > 8 hours stop
        if time.time()-then > 28800:
            print('Time is up!')
            datasetDictPath = os.path.join(dataPath, f'{experiment}DatasetDict.npy')
            np.save(datasetDictPath, datasetDicts)            
            break


# %%
if __name__ == '__main__':
    predictor = modelTools.getSegmentModel('../../models/TJ2201Split16')
    # Filter out inappropriate images
    experiment = 'TJ2201'
    finalDate = datetime.datetime(2022, 4, 8, 16, 0)

    dataPath = f'../../data/{experiment}/split16/'
    pcPath = os.path.join(dataPath, 'phaseContrast')
    compositePath = os.path.join(dataPath, 'composite')
    pcIms = os.listdir(pcPath)
    #
    compositeIms = os.listdir(compositePath)
    # Get rid of files not in appropriate well or after confluency date
    imgBases = []
    for pcFile in tqdm(pcIms):
        imgBase = fileManagement.getImageBase(pcFile)
        well = imgBase.split('_')[0]
        date = fileManagement.convertDate('_'.join(imgBase.split('_')[2:4]))
        if date < finalDate:
            imgBases.append(imgBase)
            
    random.seed(1234)
    random.shuffle(imgBases)
    # Define structure and phenotype to well

    monoPos = ['B2','B3','B4','B5','B6','C2','C3','C4','C5','C6','D2','D3','D4','D5','D6']
    monoNeg = ['E2','E3','E4','E5','E6','F2','F3','F4','F5','F6','G2','G3','G4','G5','G6']
    co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
    #
    n = 200
    imgBase = imgBases[n]

    pcFile = f'phaseContrast_{imgBase}.png'

    pcImg = imread(os.path.join(pcPath, pcFile))
    # Load and segment data
    maxCount = 5000
    #
    experiment = 'TJ2201'
    dataPath = os.path.join('../../data', experiment, 'split16')
    phenoDict = {'green': 0, 'red': 1}
    segmentExperiment(dataPath, imgBases, phenoDict, experiment, predictor)
# %%