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
from torch.utils.data import DataLoader, ConcatDataset, Subset

from skimage.measure import regionprops
from skimage.draw import polygon2mask

from src.models import trainBB
from src.data.fileManagement import getModelDetails
# %%
def getAreaEcc(polygon, imageShape):
    polyx = polygon[::2]
    polyy = polygon[1::2]
    polygonSki = list(zip(polyy, polyx))
    mask = polygon2mask(imageShape, polygonSki)
    reg = regionprops(mask.astype(np.uint8))

    area = reg[0].area
    eccentricity = reg[0].eccentricity
    
    return area, eccentricity

def filterTransfect(datasetDicts):
    newDatasetDicts = []
    c = 0
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []
        image_shape = [record['height'], record['width']]

        for annotation in record['annotations']:
            annotation['bbox'] = detectron2.structures.BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            if annotation['category_id'] == 0:
                annotation['category_id'] = 1
            else:
                annotation['category_id'] = 0

            
            segmentation = annotation['segmentation'][0]
            area, ecc = getAreaEcc(segmentation, image_shape)

            if ecc > 0.8 and annotation['category_id'] == 1:
                newAnnotations.append(annotation)
        if len(newAnnotations) > 0:
            record['annotations'] = newAnnotations
            newDatasetDicts.append(record)
            c += len(newAnnotations)
        if c > 10000:
            return newDatasetDicts
    return newDatasetDicts

def filterLPD4(datasetDicts):
    newDatasetDicts = []
    c = 0
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []
        image_shape = [record['height'], record['width']]

        for annotation in record['annotations']:
            segmentation = annotation['segmentation'][0]
            area, ecc = getAreaEcc(segmentation, image_shape)

            if ecc > 0.8:
                newAnnotations.append(annotation)
        if len(newAnnotations) > 0:
            record['annotations'] = newAnnotations
            newDatasetDicts.append(record)
            c += len(newAnnotations)

        if c > 10000:
            return newDatasetDicts
    return newDatasetDicts


# %%
datasetDicts = load_coco_json('../data/TJ2342A/TJ2342ASegmentations.json', '.')
datasetDictsTransfect = filterTransfect(datasetDicts)
# %%
datasetDictsLPD4 = np.load('../data/TJ2303-LPD4/TJ2303-LPD4DatasetDicts.npy', allow_pickle=True)
datasetDictsLPD4 = filterLPD4(datasetDictsLPD4)
# %%
c = 0
for record in datasetDictsTransfect:
    for annotation in record:
        c += len(annotation)
print(c)
# %%
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
num_epochs  = 10
modelType   = 'resnet152'
optimizer = 'sgd'
notes = 'Testing non-green transfected against LPD4'
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
experiment = 'TJ2342A'
trainLoaders, testLoaders = [], []
modelInputs['experiment'] = experiment
modelInputs['maxAmt'] = 10000
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

dataloaders, dataset_sizes = makeImageDatasets(datasetDictsTransfect, 
                                            dataPath,
                                            modelInputs,
                                            phase = ['none']
                                            )
transfectTrain = Subset(dataloaders.dataset, range(0, 7000))
transfectTest = Subset(dataloaders.dataset, range(7000, 10000))
# %%
inputs, classes = next(iter(dataloaders))
plt.imshow(inputs[19].numpy().transpose((1,2,0)))
# %%
dataPath = Path(f'../data/TJ2303-LPD4/raw/phaseContrast')
modelInputs['experiment'] = 'TJ2303-LPD4'
dataloaders, dataset_sizes = makeImageDatasets(datasetDictsLPD4, 
                                               dataPath,
                                               modelInputs,
                                               phase=['none']
                                            )
lpd4Train = Subset(dataloaders.dataset, range(0, 7000))
lpd4Test = Subset(dataloaders.dataset, range(7000, 10000))
# %%
inputs, classes = next(iter(dataloaders))
plt.imshow(inputs[19].numpy().transpose((1,2,0)))
# %%
trainLoader = DataLoader(ConcatDataset([transfectTrain, lpd4Train]),
                            batch_size = modelInputs['batch_size'],
                            shuffle = True)

testLoader = DataLoader(ConcatDataset([transfectTest, lpd4Test]),
                            batch_size = modelInputs['batch_size'],
                            shuffle = True)
# %%
inputs, classes = next(iter(trainLoader))
# %%
dataloaders = {'train': trainLoader, 'test': testLoader}
dataset_sizes = {'train': len(trainLoader.dataset), 'test': len(testLoader.dataset)}
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
modelDetailsPrint = modelTools.printModelVariables(modelInputs)


with open(resultsSaveName, 'a') as file:
    file.write(modelDetailsPrint)
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
                    num_epochs=modelInputs['num_epochs']
                    )

# %%
homePath = Path('..')
modelName = modelInputs['modelName'].split('.')[0]
probs, allLabels, scores = testBB.testModel(model, dataloaders, mode = 'test')
imgNames = ''
res = testBB.testResults(probs, allLabels, scores, imgNames, modelName)
res = vars(res)
for val in res.keys():
    if isinstance(res[val], np.ndarray):
        res[val] = res[val].tolist()
# %%
import json
json_file_loc = '../results/classificationResults/bt474Experiments.json'
with open(json_file_loc, 'r') as json_file:
    modelRes = json.load(json_file)

modelRes[modelName] = res
# %%
with open(json_file_loc, 'w') as json_file:
    json_file.write(json.dumps(modelRes))