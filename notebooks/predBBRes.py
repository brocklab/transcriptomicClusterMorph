# %%
from src.models.predBB import getModelDetails, makeImageDatasets, testModel

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from torchvision import models
import torch
import torch.nn as nn

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# %%
# First get training/testing results
homePath = Path('..')
modelId = 'classifySingleCellCrop-688020'
# %%
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

modelPath = Path.joinpath(homePath, 'models', f'{modelId}.pth')

model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(modelPath, map_location=device))
model.eval()
# %%
outPath = Path.joinpath(homePath, 'results', 'TJ2201SingleCellCrop', f'{modelId}.out')

modelDetails = getModelDetails(outPath)
experiment = modelDetails['experiment']

dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorder.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)

dataloaders, dataset_sizes = makeImageDatasets(datasetDicts,
                                               dataPath, 
                                               nIncrease    = modelDetails['nIncrease'], 
                                               maxAmt       = modelDetails['maxAmt'], 
                                               batch_size   = modelDetails['batch_size']
                                               )
# %%
probs, allLabels, scores = testModel(model, dataloaders)
# %%
fpr, tpr, _ = roc_curve(allLabels, scores[:,1])
roc_auc = roc_auc_score(allLabels, scores[:,1])
# %%
plt.plot(fpr, tpr, linewidth=3)
plt.title('')