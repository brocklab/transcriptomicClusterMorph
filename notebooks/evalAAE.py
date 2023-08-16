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
import itertools

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable

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
modelSaveName = Path(f'../models/classification/singleCellAutoencoder-{modelID}.pth')
resultsSaveName = Path(f'../results/classificationTraining/singleCellAutoencoder-{modelID}.txt')
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
'augmentation'  : None
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
# %% Parameters
n_epochs = 50
batch_size = 4
lr = .0002
b1 = 0.5
b2 = .999

latent_dim = 60
img_size = 150
channels = 1
sample_interval = 400

img_shape = (channels, img_size, img_size)

# %%
for img in tqdm(dataloaders['train']):
    break
# %%
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity
    
# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# %%
decoder.load_state_dict(torch.load('../models/testDecoder.pth'))
encoder.load_state_dict(torch.load('../models/testEncoder.pth'))

decoder.eval()
encoder.eval()
# %%
# data_loader = dataloaders['train']
# for i, (imgs, _) in tqdm(enumerate(data_loader)):
#         imgs = imgs[:, 0:1, :, :]
#         real_imgs = Variable(imgs.type(Tensor))

#         # -----------------
#         #  Train Generator
#         # -----------------

#         optimizer_G.zero_grad()

#         encoded_imgs = encoder(real_imgs)
#         decoded_imgs = decoder(encoded_imgs)
#         break
# %%
for imgs, labels in tqdm(dataloaders['train']):
    imgs = imgs[:, 0:1, :, :]
    real_imgs = Variable(imgs.type(Tensor))
    
    encodings = encoder(real_imgs)
    decodings = decoder(encodings)
    break

img0 = imgs[0][0].detach().cpu().numpy()
label0 = labels[0].detach().cpu().numpy()
decoding0 = decodings[0][0].detach().cpu().numpy()
# %%
plt.subplot(121)
plt.imshow(imgs[0][0].detach().cpu().numpy())
plt.subplot(122)
plt.imshow(decoded_imgs[0][0].detach().cpu().numpy())