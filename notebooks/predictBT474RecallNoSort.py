# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm
import argparse 

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
from skimage.measure import regionprops
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
# %% Add argparse
parser = argparse.ArgumentParser(description='Network prediction parameters')
parser.add_argument('--experiment', type = str, metavar='experiment',  help = 'Experiment to run')
parser.add_argument('--nIncrease',  type = int, metavar='nIncrease',   help = 'Increase of bounding box around cell')
parser.add_argument('--maxAmt',     type = int, metavar='maxAmt',      help = 'Max amount of cells')
parser.add_argument('--batch_size', type = int, metavar='batch_size',  help = 'Batch size')
parser.add_argument('--num_epochs', type = int, metavar='num_epochs',  help = 'Number of epochs')
parser.add_argument('--modelType',  type = str, metavar='modelType',   help = 'Type of model (resnet, vgg, etc.)')
parser.add_argument('--notes',      type = str, metavar='notes',       help = 'Notes on why experiment is being run')
parser.add_argument('--optimizer',  type = str, metavar='optimizer',   help = 'Optimizer type')
parser.add_argument('--augmentation',  type = str, metavar='augmentation',   help = 'Image adjustment (None, blackoutCell, stamp)')
parser.add_argument('--maxImgSize', type = int, metavar='maxImgSize', help = 'The final size of the image. If larger than the bounding box, pad with black, otherwise resize the image')
parser.add_argument('--nIms',       type = int, metavar='augmentation',   help = 'Number of images the initial full image was split into (experiment dependent). 20x magnification: 16, 10x magnification: 4')

# This is for running the notebook directly
args, unknown = parser.parse_known_args()

# %%
experiment  = 'TJ2321-LPD4Lin1'
nIncrease   = 20
maxAmt      = 500000000
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
optimizer = 'sgd'
notes = ''
maxImgSize = 150
nIms = 4

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
'notes'         : notes,
'optimizer'     : optimizer, 
'augmentation'  : 'None',
'testWell'      : ['B2'],
'maxImgSize'    : maxImgSize,
'nIms'          : nIms
}

argItems = vars(args)

for item, value in argItems.items():
    if value is not None:
        print(f'Replacing {item} value with {value}')
        modelInputs[item] = value
modelDetailsPrint = modelTools.printModelVariables(modelInputs)
# %%
experiments = datasetDictsGreen.keys()
loaders = []
for experiment in experiments:
    modelInputs['experiment'] = experiment
    dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

    dataloader, dataset_sizes = makeImageDatasets(datasetDictsGreen[experiment], 
                                                dataPath,
                                                modelInputs,
                                                phase = ['none']
                                                )
    loaders.append(dataloader.dataset)

# %%
sizeTrain = 0
for cellDataset in loaders[0:-1]:
    sizeTrain += len(cellDataset)
sizeTest = len(loaders[-1])

loadersTrain = loaders[0:-1]
loadersTest = [loaders[-1]]
# %%
datasetDicts = np.load('../data/TJ2303-LPD4/TJ2303-LPD4DatasetDicts.npy', allow_pickle=True)

# %%
dataPath = Path(f'../data/TJ2303-LPD4/raw/phaseContrast')
modelInputs['experiment'] = 'TJ2303-LPD4'
loadersTrain = loaders[0:-1]
loadersTest = [loaders[-1]]
from torch.utils.data import Subset
linOtherLoader, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath,
                                               modelInputs,
                                               phase=['none']
                                            )                                            
linOtherTrain = Subset(linOtherLoader.dataset, range(0, sizeTrain))
linOtherTest = Subset(linOtherLoader.dataset, range(sizeTrain, sizeTrain+sizeTest))

loadersTrain.append(linOtherTrain)
loadersTest.append(linOtherTest)
# %%
dataLoaderTrain = DataLoader(ConcatDataset(loadersTrain),
                             batch_size = modelInputs['batch_size'],
                             shuffle = True)


dataLoaderTest = DataLoader(ConcatDataset(loadersTest),
                             batch_size = modelInputs['batch_size'],
                             shuffle = True)
# %%
inputs, classes = next(iter(dataLoaderTrain))
# %%
# import matplotlib.pyplot as plt
# plt.imshow(inputs[16].numpy().transpose((1,2,0)))
# %%
dataloaders = {'train': dataLoaderTrain, 'test': dataLoaderTest}
dataset_sizes = {'train': len(dataLoaderTrain.dataset), 'test': len(dataLoaderTest.dataset)}
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
model2 = nn.DataParallel(model)

# %%
# Scheduler to update lr
# Every 7 epochs the learning rate is multiplied by gamma
setp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, 
                    criterion, 
                    optimizer, 
                    setp_lr_scheduler,
                    dataloaders, 
                    dataset_sizes, 
                    modelSaveName,
                    resultsSaveName,
                    num_epochs=num_epochs
                    )
# %%
