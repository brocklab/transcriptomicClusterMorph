# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm
import argparse 
import pickle 

from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
import detectron2

from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.models import testBB
from src.data.fileManagement import convertDate
from src.models import modelTools
from pathlib import Path
import numpy as np
import time
import sys
import datetime
import matplotlib.pyplot as plt

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from src.models import trainBB
from src.data.fileManagement import getModelDetails

# %%
def getGreenRecord(datasetDicts, datasetDictsGreen = []):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []

        for annotation in record['annotations']:
            annotation['bbox'] = detectron2.structures.BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            # if annotation['category_id'] == 1:
            #     newAnnotations.append(annotation)
        # if len(newAnnotations) > 0:
        #     record['annotations'] = newAnnotations
        #     datasetDictsGreen.append(record)
    return datasetDicts
# for record in tqdm(datasetDictsTreat):
#     for cell in record['annotations']:
#         cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
#         cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%
datasetDictsGreen = {}

datasetDicts = load_coco_json('../../../data/TJ2342A/TJ2342ASegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2342A'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442B/TJ2442BSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442B'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442C/TJ2442CSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442C'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442D/TJ2442DSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442D'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442E/TJ2442ESegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442E'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442F/TJ2442FSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442F'] = getGreenRecord(datasetDicts, [])
# %% Try to find dead cells
from skimage.measure import regionprops
from skimage.draw import polygon2mask

def getAreaEcc(polygon, imageShape):
    polyx = polygon[::2]
    polyy = polygon[1::2]
    polygonSki = list(zip(polyy, polyx))
    mask = polygon2mask(imageShape, polygonSki)
    reg = regionprops(mask.astype(np.uint8))

    if len(reg) > 0:
        area = reg[0].area
        eccentricity = reg[0].eccentricity
    
    else:
        area = 0
        eccentricity = 0

    return area, eccentricity
for experiment in datasetDictsGreen:
    datasetDicts = datasetDictsGreen[experiment]
    nAnnos = 0
    for record in datasetDicts:
        nAnnos += len(record['annotations'])
    print(f'{experiment} had {nAnnos} cells identified')

# %%
allArea, allEcc, imgNames, allPoly = [], [], [], []

# %%
for experiment in datasetDictsGreen.keys():
    datasetDicts = datasetDictsGreen[experiment]
    print(experiment)
    nAnnos = 0
    for record in tqdm(datasetDicts):
        image_shape = [record['height'], record['width']]
        newAnnotations = []
        for annotation in record['annotations']:
            segmentation = annotation['segmentation'][0]

            area, ecc = getAreaEcc(segmentation, image_shape)

            if ecc > 0.8:
                newAnnotations.append(annotation)
            allArea.append(area)
            allEcc.append(ecc)
            allPoly.append(segmentation)
            imgNames.append(record['file_name'])
        nAnnos += len(newAnnotations)
        record['annotations'] = newAnnotations

    print(f'{experiment} had {nAnnos} cells identified')
# %%
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.datasets as datasets
    
def getCells(datasetDict):
    return datasetDict

for experiment in datasetDictsGreen.keys():
    fileName = f'../../../data/{experiment}/{experiment}SegmentationsGreenFiltered.json'
    datasetDicts = datasetDictsGreen[experiment]
    inputs = [datasetDicts]
    if 'cellMorph' in DatasetCatalog:
        DatasetCatalog.remove('cellMorph')
        MetadataCatalog.remove('cellMorph')
    DatasetCatalog.register("cellMorph", lambda x=inputs: getCells(inputs[0]))
    MetadataCatalog.get("cellMorph").set(thing_classes=["cell"])
    datasets.convert_to_coco_json('cellMorph', output_file=fileName, allow_cached=False)

# %%
def makeDatasets(modelName, homePath = '../../../'):
    modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
    outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
    if not outPath.exists():
        outPath = Path(str(outPath).replace('.out', '.txt'))
    assert outPath.exists(), outPath
    modelInputs = getModelDetails(outPath)
    experiments = datasetDictsGreen.keys()
    loaders = []
    for experiment in experiments:
        modelInputs['experiment'] = experiment
        dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

        dataloader, dataset_sizes = makeImageDatasets(datasetDictsGreen[experiment], 
                                                    dataPath,
                                                    modelInputs,
                                                    phase = ['train']
                                                    )
        loaders.append(dataloader.dataset)

    sizeTrain = 0
    for cellDataset in loaders[0:-1]:
        sizeTrain += len(cellDataset)
    sizeTest = len(loaders[-1])

    loadersTrain = loaders[0:-1]
    loadersTest = [loaders[-1]]

    dataLoaderTrain = DataLoader(ConcatDataset(loadersTrain),
                                batch_size = modelInputs['batch_size'],
                                shuffle = True)


    dataLoaderTest = DataLoader(ConcatDataset(loadersTest),
                                batch_size = modelInputs['batch_size'],
                                shuffle = True)

    dataloaders = {'train': dataLoaderTrain, 'test': dataLoaderTest}
    dataset_sizes = {'train': len(dataLoaderTrain.dataset), 'test': len(dataLoaderTest.dataset)}

    return dataloaders


# %%
homePath = Path('../../../')
resultsFile = homePath / 'results' / 'classificationResults' / 'modelResultsCoCulture.pickle'
if resultsFile.exists():
    modelRes = pickle.load(open(resultsFile, "rb"))
else:
    modelRes = {}

modelNames = [
            #   'classifySingleCellCrop-714689',
            #   'classifySingleCellCrop-713279', 
            #   'classifySingleCellCrop-709125',
              'classifySingleCellCrop-1707264894', # 65 px 
              'classifySingleCellCrop-1707668614', # 25 px
              'classifySingleCellCrop-1707714016', # 00 px
              'classifySingleCellCrop-1709261519', # 55 px
              'classifySingleCellCrop-1709418455', # 15 px
              'classifySingleCellCrop-1709372973', # 45 px
              'classifySingleCellCrop-1709327523'  # 35 px
             ]
for modelName in modelNames:
    if modelName not in modelRes.keys():
        print(modelName)
        probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts)
        modelRes[modelName] = testBB.testResults(probs, allLabels, scores, imgNames, modelName)

pickle.dump(modelRes, open(resultsFile, "wb"))