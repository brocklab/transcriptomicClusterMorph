# %% [markdown]
"""
This will classify cells using mask rcnn 
"""
# %%
from src.models.modelTools import getSegmentModel
from src.visualization.segmentationVis import viewPredictorResult
from src.data.imageProcessing import findFluorescenceColor

import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2 

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
import torch
# %%
experiment = 'TJ2201'
# %%
homePath = Path('..')
dataPath = homePath / Path(f'/data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDict.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %%
for seg in tqdm(datasetDicts):

    filePath = Path(seg['file_name'])
    filePath = '../' / Path(*filePath.parts[2:])
    seg['file_name'] = str(filePath)
    assert filePath.exists()
# %% Balance dataset dicts so that there is a roughly equal number of cells

nCellsMax = 15000
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDictTrain = []
totalPhenoCount = {0: 0, 1: 0}
imgPhenoCounts = {}
for idx, seg in enumerate(datasetDicts):
    nCells = len(seg['annotations'])
    well = seg['file_name'].split('_')[1]
    if nCells == 0:
        continue
    if well not in co:
        continue
    cts = [0, 0]
    for cell in seg['annotations']:
        cts[cell['category_id']] += 1
    if cts[0] == cts[1] or cts[0] > cts[1]:
       totalPhenoCount[0] += cts[0]
       totalPhenoCount[1] += cts[1]
       
       datasetDictTrain.append(seg)
for idx, seg in enumerate(datasetDicts):
    nCells = len(seg['annotations'])
    well = seg['file_name'].split('_')[1]
    if nCells == 0:
        continue
    if well not in co:
        continue
    cts = [0, 0]
    for cell in seg['annotations']:
        cts[cell['category_id']] += 1
    if cts[1] > cts[0]:
        totalPhenoCount[0] += cts[0]
        totalPhenoCount[1] += cts[1]
        datasetDictTrain.append(seg)
    if totalPhenoCount[1] > totalPhenoCount[0]:
        break
# %%
random.seed(1234)
random.shuffle(datasetDictTrain)

nTrain = int(.9*len(datasetDictTrain))
ddTrain = datasetDictTrain[0:nTrain]
ddTest = datasetDictTrain[nTrain:]

newPhenoCount = [0, 0]
for seg in ddTrain:
    for cell in seg['annotations']:
        newPhenoCount[cell['category_id']] += 1 
print(newPhenoCount)
# %%
def registerDatasetDict(phase):
 
    random.seed(1234)
    random.shuffle(datasetDictTrain)

    nTrain = int(.9*len(datasetDictTrain))

    ddTrain = datasetDictTrain[0:nTrain]
    ddTest = datasetDictTrain[nTrain:]

    if phase == 'train':
        return ddTrain
    elif phase == 'test':
        return ddTest
    
name = 'cocultureClassifier'
imgType = 'phaseContrast'
trainName = f'{name}_train'
testName = f'{name}_test'
if trainName in DatasetCatalog:
    DatasetCatalog.remove(trainName)
    MetadataCatalog.remove(trainName)
    print('Removing training')
if testName in DatasetCatalog:
    DatasetCatalog.remove(testName)
    MetadataCatalog.remove(testName)
    print('Removing testing')
inputs = [experiment, imgType, 'train']

DatasetCatalog.register(trainName, lambda x = inputs: registerDatasetDict('train'))
MetadataCatalog.get(trainName).set(thing_classes=["esamPos", "esamNeg"])

DatasetCatalog.register(testName, lambda x=inputs: registerDatasetDict('test'))
MetadataCatalog.get(testName).set(thing_classes=["esamPos", "esamNeg"])

cell_metadata = MetadataCatalog.get('cellMorph_train')
# %%
modelPath = homePath / 'models' / 'segmentation' / 'cocultureClassify2'
classPath = homePath / 'models/segmentation/TJ2201Split16'
cocultureClass = getSegmentModel(modelPath, numClasses=3)
pred2 = getSegmentModel(classPath, numClasses=1)
# %%
ddTrain = registerDatasetDict('train')
ddTest = registerDatasetDict('test')
# %%
idx = 3
img = imread(ddTest[idx]['file_name'])
imgComposite = imread(ddTest[idx]['file_name'].replace('phaseContrast', 'composite'))
# %%
outputsCo = cocultureClass(img)
outputsTJ = pred2(img)

print(len(outputsCo['instances']))
print(len(outputsTJ['instances']))
print(len(ddTest[idx]['annotations']))
# %%
viewPredictorResult(cocultureClass, img)
plt.imshow(imgComposite)
# %%
# 0 - esam(+) - green fluorescence
# 1 - esam(-) - red fluorescence
colorDict = {'green': 0, 'red': 0}
amtPred = {}
amtActual = {}
imgAcc = {}
totalCorrect, total = 0, 0
for seg in (ddTrain):
    filePath = Path(seg['file_name'])
    fileName = filePath.parts[-1]
    img = imread(filePath)
    imgComposite = imread(seg['file_name'].replace('phaseContrast', 'composite'))
    predOutputs = cocultureClass(img)
    amtPred[fileName] = [0, 0]
    amtActual[fileName] = [0, 0]

    masks = predOutputs['instances']._fields['pred_masks']
    predClasses = predOutputs['instances']._fields['pred_classes']
    if len(predClasses) == 0:
        continue
    imgCorrect, imgTotal = 0, 0
    for mask, predClass in zip(masks, predClasses):
        cellColor = findFluorescenceColor(imgComposite, np.array(masks[0]))
        if cellColor == 'NaN':
            continue
        trueClass = colorDict[cellColor]
        predClass = int(predClass)

        amtPred[fileName][predClass] += 1
        amtActual[fileName][trueClass] += 1
        
        if trueClass == predClass:
            imgCorrect += 1
            totalCorrect += 1
        total += 1
        imgTotal += 1
    
    if imgTotal > 0:
        imgAcc[fileName] = [imgCorrect, imgTotal]
        print(f'Image correct: {imgCorrect/imgTotal:0.2} \t {imgCorrect}/{imgTotal}, \t {totalCorrect/total}')
#%%