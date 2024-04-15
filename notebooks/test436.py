# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
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
datasetDicts = load_coco_json(f'../data/{experiment}/{experiment}Segmentations.json', '.')
datasetDicts = convertRecords(datasetDicts)
# %%
modelName = 'classifySingleCellCrop-1713070074'
homePath = Path('../')
probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts, mode = 'test')
# %%
res = testBB.testResults(probs, allLabels, scores, imgNames, modelName)
# %%
experiment  = 'TJ2453-436Co'
datasetDicts = load_coco_json(f'../data/{experiment}/{experiment}SegmentationsFiltered.json', '.')
datasetDicts = convertRecords(datasetDicts)
# %%
modelName = 'classifySingleCellCrop-1713133318'
homePath = Path('../')
probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts, mode = 'test')
# %%
resFiltered = testBB.testResults(probs, allLabels, scores, imgNames, modelName)
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
