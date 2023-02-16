# %% [markdown]
"""
This will classify cells using mask rcnn 
"""
# %%
import numpy as np
from pathlib import Path
from tqdm import tqdm

from detectron2.data import MetadataCatalog, DatasetCatalog

# %%
experiment = 'TJ2201'
# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorder.npy')
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

def 
name = 'cocultureClassifier'
imgType = 'phaseContrast'
trainName = f'{name}_train'
testName = f'{name}_test'
if trainName in DatasetCatalog:
    DatasetCatalog.remove(trainName)
    print('Removing training')
if testName in DatasetCatalog:
    DatasetCatalog.remove(testName)
    print('Removing testing')
inputs = [experiment, imgType, 'train']

DatasetCatalog.register(trainName, datasetDictTrain)
MetadataCatalog.get(trainName).set(thing_classes=["cell"])

# DatasetCatalog.register(testName, lambda x=inputs: cellpose2Detectron(inputs[0], inputs[1], 'test'))
# MetadataCatalog.get(testName).set(thing_classes=["cell"])
