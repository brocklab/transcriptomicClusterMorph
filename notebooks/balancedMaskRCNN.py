# %% [markdown]
"""
This will classify cells using mask rcnn 
"""
# %%
import numpy as np
from pathlib import Path
from tqdm import tqdm
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
nCells = 15000
datasetDictTrain = []
phenoCount = {0: 0, 1: 0}
for seg in datasetDicts:
    nCells = len(seg['annotations'])
    if nCells > 0:
        pass