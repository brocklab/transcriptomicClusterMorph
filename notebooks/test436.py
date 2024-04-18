# %%
from src.data.fileManagement import getModelDetails

from src.models import testBB
from src.models import modelTools
from pathlib import Path
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm 

from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim

from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
# %%
def convertRecords(datasetDicts):
    newDatasetDicts = []
    for record in tqdm(datasetDicts):
        well = record['file_name'].split('_')[1]
        if well.endswith('10') or well.endswith('11'):
            continue
        record = record.copy()

        for annotation in record['annotations']:
            annotation['bbox'] = BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
        newDatasetDicts.append(record)
    return newDatasetDicts
# %%
experiment  = 'TJ2453-436Co'
datasetDicts = load_coco_json(f'../data/{experiment}/{experiment}SegmentationsFiltered.json', '.')
datasetDicts = convertRecords(datasetDicts)
# %%
modelNames = [
            'classifySingleCellCrop-1713133318',
            'classifySingleCellCrop-1713198231',
            'classifySingleCellCrop-1713219654',
            'classifySingleCellCrop-1713276348',
            'classifySingleCellCrop-1713297660',
            'classifySingleCellCrop-1713369737',
            'classifySingleCellCrop-1713398690',
            'classifySingleCellCrop-1713420430'
]
resDict = {}
homePath = Path('../')
for modelName in modelNames:
    probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts, mode = 'test')
    res = testBB.testResults(probs, allLabels, scores, imgNames, modelName)
    resDict[modelName] = res
# %%
oldModelNames = list(resDict.keys())
for modelName in oldModelNames:
    if modelName not in modelNames:
        resDict.pop(modelName)
# %%
aucs, nIncreases = [], []
for modelName in modelNames:
    modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
    outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
    if not outPath.exists():
        outPath = Path(str(outPath).replace('.out', '.txt'))
    assert outPath.exists(), outPath
    modelDetails = getModelDetails(outPath)
    if modelDetails['nIncrease'] > 100:
        continue
    aucs.append(resDict[modelName].auc)
    nIncreases.append(modelDetails['nIncrease'])
# %%
iA = list(zip(nIncreases, aucs))
iA.sort()
nIncreases = [i for i, a in iA]
aucs = [a for i, a in iA]


plt.figure()
plt.figure(figsize=(8,5))
plt.rcParams.update({'font.size': 17})
plt.scatter(nIncreases, aucs, s = 100)
plt.plot(nIncreases, aucs)
plt.xticks(nIncreases)
plt.xlabel('Pixel Increase')
plt.ylabel('AUC')
plt.savefig('../figures/publication/results/increasingBBSubpop436.png', dpi = 500, bbox_inches = 'tight')

# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
auc = res.auc
plotLabel = f'BB increase AUC = {auc:0.2f}'
plt.plot(res.fpr, res.tpr, 
         label=f'All cells AUC={res.auc:0.2f}', linewidth=3)
plt.plot(resFiltered.fpr, resFiltered.tpr, 
         label=f'Better labeling AUC={resFiltered.auc:0.2f}', 
         linewidth=3)
plt.legend(fontsize = 10, loc = 'lower right')
plt.title('MDA-MB-436 Classification')
# %%
