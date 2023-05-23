# %%
from src.data import fileManagement
from src.models.trainBB import train_model, getTFModel
from src.models import modelTools

from pathlib import Path
import datetime
import random
import numpy as np
from skimage.io import imread
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim
# %%
experiment  = 'TJ2201'
batch_size  = 32
num_epochs  = 32
modelType   = 'resnet152'
optimizer = 'sgd'
notes = 'Run on split image coculture wells only'

modelID, idSource = modelTools.getModelID(sys.argv)
modelSaveName = Path(f'../models/classification/classifySplitCoculture-{modelID}.pth')
resultsSaveName = Path(f'../results/classificationTraining/classifySplitCoculture-{modelID}.txt')
modelInputs = {

'experiment'    : experiment, 
'batch_size'    : batch_size,
'num_epochs'    : num_epochs,
'modelType'     : modelType,
'modelName'     : modelSaveName.parts[-1],
'modelIDSource' : idSource,
'notes'         : notes,
'optimizer'     : optimizer
}
# %% Filter 
dataPath = Path('../data/TJ2201/split16/phaseContrast')
allImages = list(dataPath.glob('*'))
imagePaths = []
for image in allImages:
    image = str(image)
    date = '_'.join(image.split('_')[3:5])
    date = fileManagement.convertDate(date)
    if date < datetime.datetime(2022, 4, 8, 16, 0):
        imagePaths.append(image)
# %%
wellDict = {
    'esamPos':[
        'B2', 'B3', 'B4', 'B5', 'B6',
        'C2', 'C3', 'C4', 'C5', 'C6',
        'D2', 'D3', 'D4', 'D5', 'D6'
    ],
    'esamNeg':[
        'E2', 'E3', 'E4', 'E5', 'E6',
        'F2', 'F3', 'F4', 'F5', 'F6',
        'G2', 'G3', 'G4', 'G5', 'G6',
    ]
}
# %% Data loader
class esamCocultureLoader(Dataset):

    def __init__(self, imagePaths, wellDict, transforms, phase, seed = 1234):
        testWells = ['B6', 'G6']
        self.transforms = transforms
        self.imagePaths = imagePaths
        self.wellDict = wellDict
        self.seed = 1234
        self.phase = phase

        phenotypes, imagesCo = [], []
        testPhenotypes, testImagesCo = [], []
        for imagePath in imagePaths:
            well = imagePath.split('_')[1]

            if well in testWells and well in wellDict['esamPos']:
                testPhenotypes.append(0)
                testImagesCo.append(imagePath)
                continue
            if well in testWells and well in wellDict['esamNeg']:
                testPhenotypes.append(1)
                testImagesCo.append(imagePath)
                continue           
                
            if well in wellDict['esamPos']:
                phenotypes.append(0)
                imagesCo.append(imagePath)
            elif well in wellDict['esamNeg']:
                phenotypes.append(1)
                imagesCo.append(imagePath)



        if phase == 'test':
            self.phenos = testPhenotypes
            self.imagePaths = testImagesCo
        elif phase == 'train':
            # Balance
            imagesCo, phenotypes = self.shuffleLists([imagesCo, phenotypes], seed = seed)
            _, cts = np.unique(phenotypes, return_counts=True)
            minCts = np.min(cts)
            phenoCount = {0: 0, 1: 0}
            phenosFinal, imagesCoFinal = [], []
            for img, pheno in zip(imagesCo, phenotypes):
                if phenoCount[pheno] <= minCts:
                    phenosFinal.append(pheno)
                    imagesCoFinal.append(img)
            self.phenos = phenosFinal
            self.imagePaths = imagesCoFinal


    def __getitem__(self, idx):
        img = imread(self.imagePaths[idx])
        if self.transforms:
            img = Image.fromarray(np.uint8(img*255))
            img = self.transforms(img)
        pheno = self.phenos[idx]
        return img, pheno

    def __len__(self):
        return len(self.phenos)

    @staticmethod
    def shuffleLists(l, seed=1234):
        random.seed(seed)
        l = list(zip(*l))
        random.shuffle(l)

        return [np.array(itm, dtype='object') for itm in list(zip(*l))]
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
        transforms.Normalize(mean, std)
    ])
}

phase = ['train', 'test']
image_datasets = {x: esamCocultureLoader(imagePaths, wellDict, data_transforms[x], phase=x) 
                  for x in phase}
dataset_sizes = {x: len(image_datasets[x]) for x in phase}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=modelInputs['batch_size'], shuffle=True)
                    for x in phase}
# %%
inputs, classes = next(iter(dataloaders['train']))
print(classes)
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = getTFModel(modelInputs['modelType'])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# %%
modelDetailsPrint = modelTools.printModelVariables(modelInputs)

with open(resultsSaveName, 'a') as file:
    file.write(modelDetailsPrint)

# Scheduler to update lr
# Every 7 epochs the learning rate is multiplied by gamma
setp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, 
                    criterion, 
                    optimizer, 
                    setp_lr_scheduler, 
                    dataloaders, 
                    dataset_sizes, 
                    modelSaveName,
                    resultsSaveName,
                    num_epochs=num_epochs
                    )
# %%
