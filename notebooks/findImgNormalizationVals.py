# %%
from src.models.trainBB import makeImageDatasets, train_model
from src.models import modelTools
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

import torch
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
# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorder.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %%
dataloader, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath, 
                                               phase = ['none'],
                                               nIncrease=nIncrease, 
                                               maxAmt = None, 
                                               batch_size=batch_size
                                               )
# %%
nChannels = 3
mean = torch.zeros(nChannels)
std = torch.zeros(nChannels)
c = 0
for inputs, _labels in tqdm(dataloader):
    for i in range(nChannels):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()
    c += 1
mean.div_(c)
std.div_( c)
print(f'mean: {mean}')
print(f'std: {std}')