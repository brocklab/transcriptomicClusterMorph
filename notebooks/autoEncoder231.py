# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.data.fileManagement import convertDate
from src.models import modelTools
from pathlib import Path
import numpy as np
import time
import sys
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim

# %%
experiment  = 'TJ2201'
nIncrease   = 25
maxAmt      = 10000
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
optimizer = 'sgd'
notes = 'Full run without blackout'

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
'optimizer'     : optimizer
}

# %%

# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
datasetDicts = [record for record in datasetDicts if len(record['annotations']) > 0]
# %%
# Check size
wellSize = {}
for seg in datasetDicts:
    well = seg['file_name'].split('_')[1]
    if well not in wellSize.keys():
        wellSize[well] = 0
    wellSize[well] += len(seg['annotations'])

sum(list(wellSize.values()))
# %%
dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath,
                                               modelInputs
                                            )
np.unique(dataloaders['train'].dataset.phenotypes, return_counts=True)
# %%
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # nimages, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1), # N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),# N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8),# N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64, out_channels=32, kernel_size=8), # N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels=16, kernel_size=4, stride=2, padding=1, output_padding=1), # N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 16, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0), # N, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encoding(self, x):
        return self.encoder(x)
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNNAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)
model.cuda()
model.to('cuda')

# %%
num_epochs = 15
outputs = []
allLoss = []
for epoch in range(num_epochs):
    for (img, label) in tqdm(dataloaders['train']):
        img = img[:,0:1,:,:]
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# %%
torch.save(model.state_dict(), './testAutoencoder.py')
# %%
savedModel = torch.load('./testAutoencoder.py', map_location=device)
model.load_state_dict(savedModel)
# %%
img, labels = next(iter(dataloaders['test']))

img = img[:,0:1,:,:]
img = img.to(device)
# %%
res = model(img)
# %%
plt.subplot(121)
plt.imshow(img[0][0].cpu().detach().numpy())
plt.title('Original')
plt.subplot(122)
plt.imshow(res[0][0].cpu().detach().numpy())
plt.title('Reconstructed')
# %%
encode = model.encoding(img)
# %%
