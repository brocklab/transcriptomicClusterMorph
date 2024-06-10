# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm

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
aucs = [0.95, .8, .75]
pcc = [0.96, 0.79, -0.03]

plt.scatter(aucs, pcc)
# %%
def getGreenRecord(datasetDicts, datasetDictsGreen = []):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []

        for annotation in record['annotations']:
            annotation['bbox'] = detectron2.structures.BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            if annotation['category_id'] == 1:
                newAnnotations.append(annotation)
        if len(newAnnotations) > 0:
            record['annotations'] = newAnnotations
            datasetDictsGreen.append(record)
    return datasetDictsGreen
# for record in tqdm(datasetDictsTreat):
#     for cell in record['annotations']:
#         cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
#         cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%
datasetDictsGreen = {}

datasetDicts = load_coco_json('../data/TJ2342A/TJ2342ASegmentations.json', '.')
datasetDictsGreen['TJ2342A'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442D/TJ2442DSegmentations.json', '.')
datasetDictsGreen['TJ2442D'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442E/TJ2442ESegmentations.json', '.')
datasetDictsGreen['TJ2442E'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442F/TJ2442FSegmentations.json', '.')
datasetDictsGreen['TJ2442F'] = getGreenRecord(datasetDicts, [])
# %%
import json
experimentParamsLoc = '/home/user/work/cellMorph/data/experimentParams.json'
with open(experimentParamsLoc, 'r') as json_file:
    experimentParams = json.load(json_file)
for experiment in datasetDictsGreen.keys():
    experimentParams[experiment] = experimentParams['TJ2303-LPD4'].copy()
experimentJson = json.dumps(experimentParams)
with open(experimentParamsLoc, 'w') as json_file:
    json_file.write(experimentJson)
# %% Try to find dead cells
from skimage.measure import label, regionprops, regionprops_table
from skimage.draw import polygon2mask

def getAreaEcc(polygon, imageShape):
    polyx = polygon[::2]
    polyy = polygon[1::2]
    polygonSki = list(zip(polyy, polyx))
    mask = polygon2mask(imageShape, polygonSki)
    reg = regionprops(mask.astype(np.uint8))

    area = reg[0].area
    eccentricity = reg[0].eccentricity
    
    return area, eccentricity

allArea, allEcc, imgNames, allPoly = [], [], [], []
for experiment in datasetDictsGreen:
    datasetDicts = datasetDictsGreen[experiment]
    nAnnos = 0
    for record in datasetDicts:
        nAnnos += len(record['annotations'])
    print(f'{experiment} had {nAnnos} cells identified')
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

            if area < 1000 and ecc < 0.3:
                newAnnotations.append(annotation)
            allArea.append(area)
            allEcc.append(ecc)
            allPoly.append(segmentation)
            imgNames.append(record['file_name'])
        nAnnos += len(newAnnotations)
        record['annotations'] = newAnnotations

    print(f'{experiment} had {nAnnos} cells identified')
# %%
plt.scatter(allArea, allEcc, s = 3)
plt.xlabel('Area')
plt.ylabel('Eccentricity')
# %% Find relevant images
# import pandas as pd
# dfCell = pd.DataFrame([allArea, allEcc, allPoly, imgNames]).T
# dfCell.columns = ['area', 'eccentricity', 'polygon', 'imageNames']

# isSmall = np.logical_and(dfCell['area'] < 1000, dfCell['eccentricity'] < 0.3)
# dfSmall = dfCell.loc[isSmall]

# imgSmall = './../data/TJ2342A/raw/phaseContrast/phaseContrast_D3_1_2024y03m01d_08h23m.png'
# poly = dfSmall['polygon'][0]
# polyx = poly[::2]
# polyy = poly[1::2]
# plt.imshow(imread(imgSmall), cmap = 'gray')
# plt.plot(polyx, polyy)
# %%
# dfBig = dfCell.loc[dfCell['eccentricity'] > 0.3]

# imgBig = './../data/TJ2342A/raw/phaseContrast/phaseContrast_G5_1_2024y02m28d_22h23m.png'

# poly = dfBig['polygon'].iloc[0]
# polyx = poly[::2]
# polyy = poly[1::2]
# plt.imshow(imread(imgBig), cmap = 'gray')
# plt.plot(polyx, polyy)
# plt.savefig('./testImg', dpi = 1000)

# %%
experiment  = 'TJ2302'
nIncrease   = 20
maxAmt      = 9e9
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
notes = 'Full test with Adam'

modelID, idSource = modelTools.getModelID(sys.argv)
modelSaveName = Path(f'../models/classification/classifySingleCellCrop-{modelID}.pth')
resultsSaveName = Path(f'../results/classificationTraining/classifySingleCellCrop-{modelID}.txt')
modelInputs = {

'experiment'    : experiment, 
'nIncrease'     : nIncrease,
'maxAmt'        : maxAmt,
'batch_size'    : batch_size,
'num_epochs'    : num_epochs,
'modelType'     : modelType,
'modelName'     : modelSaveName.parts[-1],
'modelIDSource' : idSource,
'notes'         : notes

}

# %%
experiments = datasetDictsGreen.keys()
trainLoaders, testLoaders = [], []
for experiment in experiments:
    modelInputs['experiment'] = experiment
    dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

    dataloaders, dataset_sizes = makeImageDatasets(datasetDictsGreen[experiment], 
                                                dataPath,
                                                modelInputs
                                                )
    
    trainLoaders.append(dataloaders['train'].dataset)
    testLoaders.append(dataloaders['test'].dataset)
# %%
trainLoaders = DataLoader(ConcatDataset(trainLoaders),
                             batch_size = modelInputs['batch_size'],
                             shuffle = True)

testLoaders = DataLoader(ConcatDataset(testLoaders),
                             batch_size = modelInputs['batch_size'],
                             shuffle = True)
# %%
inputs, classes = next(iter(trainLoaders))
# %%
plt.imshow(inputs[18].numpy().transpose((1,2,0)))
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not modelSaveName.parent.exists():
    raise NotADirectoryError('Model directory not found')

model = getTFModel(modelInputs['modelType'])
model.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
modelName = 'classifySingleCellCrop-1700026902'
modelName = 'classifySingleCellCrop-1700187095'
homePath = Path('../')
modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
if not outPath.exists():
    outPath = Path(str(outPath).replace('.out', '.txt'))
assert outPath.exists(), outPath
modelInputs = getModelDetails(outPath)
modelInputs['augmentation'] = None
print(modelInputs)
model = trainBB.getTFModel(modelInputs['modelType'], modelPath)

dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

modelInputs['experiment'] = 'TJ2310'
# %%
allPreds = []
model.to(device)
for inputs, labels in tqdm(trainLoaders):
    inputs = inputs.float()
    inputs = inputs.to(device)
    labels = labels.to(device)


    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    print(preds)
    allPreds.append(preds)

# %%
print(allPreds)
# %%
