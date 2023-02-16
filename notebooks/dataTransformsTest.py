# %%
from src.models.trainBB import makeImageDatasets, train_model
from src.models import modelTools
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

import torch
from torchvision import transforms
# %%
# %%
experiment  = 'TJ2201'
nIncrease   = 65
maxAmt      = 15000
batch_size  = 40
num_epochs  = 40
modelType   = 'resnet152'
notes = 'Run only on coculture wells'

modelID, idSource = modelTools.getModelID(sys.argv)
modelSaveName = Path(f'../models/classifySingleCellCrop-{modelID}.pth')

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

modelTools.printModelVariables(modelInputs)
# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorder.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
# %%
mean = np.array([0.4840, 0.4840, 0.4840])
std = np.array([0.1047, 0.1047, 0.1047])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        # transforms.Resize(356),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'none': transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
}
# %%
dataloader, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath, 
                                               data_transforms = data_transforms,
                                               phase = ['none'],
                                               nIncrease=nIncrease, 
                                               maxAmt = None, 
                                               batch_size=batch_size
                                               )
# %%
for inputs, _labels in tqdm(dataloader):
 
    break
# %%
